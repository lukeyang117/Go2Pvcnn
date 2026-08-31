# Parallelism AMP PPO 对齐与 1024 真实测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Parallelism AMP 的 Actor/base critic/AMP critic 更新行为与当前普通 PPO 对齐，并提供可显式运行的 1024 环境真实 4-iteration 验收测试。

**Architecture:** 保留 `ParallelismAMPPPO` 的双 GAE、AMP replay 和独立 D；共享 `PPO` 的 adaptive-KL、clipped value loss、全局梯度裁剪、学习率和 std 下限语义。AMP critic 使用现有 critic observation 加 `amp_active/history_ratio`，其 loss 只由 active 行贡献；pure PPO 与 distillation 入口不改变。

**Tech Stack:** Python 3.10, PyTorch, Isaac Lab/Isaac Sim, pytest, TensorBoard event files, Bash launcher。

## Global Constraints

- AMP rollout 固定 `num_steps_per_env=40`，history window 固定 24 帧，transition 增量必须保持 batch Torch/GPU 路径。
- PPO 共享配置为 `num_learning_epochs=5`、`num_mini_batches=4`、`learning_rate=1e-3`、`clip_param=0.2`、`gamma=0.99`、`lam=0.95`、`value_loss_coef=1.0`、`amp_value_loss_coef=1.0`、`entropy_coef=0.01`、`max_grad_norm=1.0`、`use_clipped_value_loss=True`、`schedule="adaptive"`、`desired_kl=0.01`。
- AMP actor guidance 在 iteration `<500` 为 0，`500..600` 线性升到 0.1；AMP critic 与 D 从第 0 iteration 训练。
- standstill/inactive transition 的 AMP reward、AMP GAE、AMP value loss 和 D 样本权重必须为 0。
- 旧 pure PPO 与 distillation 实验的类名、配置和调用路径保持不变。
- 1024 smoke 必须真实启动 Isaac Lab、恢复指定 checkpoint、完成 4 次 PPO iteration 并以退出码 0 结束；普通 pytest 不自动启动该测试。

## 文件边界

- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/parallelism_amp_ppo.py`：复用普通 PPO 的 KL/value/std/metrics 语义，保留 AMP 双 value、replay 和 D。
- Modify: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_update.py`：增加 adaptive-KL、AMP clipped value、std clipping 和 finite gradient 行为测试。
- Modify: `Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py`：加入显式 `RUN_REAL_AMP_1024` 门控、TensorBoard/日志检查、GPU/耗时/显存结果记录。
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_amp_training_smoke_probe.py`：验证 1024 smoke 命令契约和 opt-in 保护。
- Modify: `Go2Pvcnn/agent/train_cfg.py`：仅在 AMP 配置中显式写入共享 PPO 参数和两个 value coefficient（若现有值已继承则补测试，不改旧配置）。
- Create: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_ppo_alignment.py`：独立的 PPO 对齐回归测试。
- Create: `docs/superpowers/plans/2026-08-31-parallelism-amp-ppo-alignment.md`：本实施计划。

### Task 1: 为 PPO 对齐行为写失败测试

**Files:**
- Create: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_ppo_alignment.py`
- Modify: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_update.py`

**Interfaces:**
- Consumes: `ParallelismAMPPPO`, `AmpActorCriticCNN`, `ParallelismAMPStorage`。
- Produces: executable tests proving AMP update uses `storage.mu/sigma`, shared adaptive learning rate, clipped AMP value target, and `actor_critic.clip_std()`.

- [ ] **Step 1: Write failing tests**

```python
def test_amp_update_uses_adaptive_kl_and_shared_learning_rate(monkeypatch):
    alg, storage = make_algorithm_and_storage()
    storage.mu.fill_(0.0)
    storage.sigma.fill_(1.0)
    monkeypatch.setattr(alg.actor_critic, "clip_std", lambda min: setattr(alg, "std_clipped", min))
    metrics = alg.update()
    assert metrics["approx_kl"] >= 0.0
    assert alg.optimizer.param_groups[0]["lr"] == alg.learning_rate
    assert hasattr(alg, "std_clipped")


def test_amp_value_loss_is_clipped_and_ignores_inactive_rows():
    alg, storage = make_algorithm_and_storage()
    storage.amp_active.zero_()
    storage.amp_active[0, 0] = 1.0
    storage.amp_values.fill_(100.0)
    storage.amp_returns.zero_()
    metrics = alg.update()
    assert torch.isfinite(torch.tensor(metrics["amp_value_loss"]))
    assert metrics["amp_active_count"] == 1


def test_amp_update_reports_finite_shared_gradient_norm():
    alg, _ = make_algorithm_and_storage()
    metrics = alg.update()
    assert torch.isfinite(torch.tensor(metrics["actor_critic_grad_norm"]))
    assert metrics["actor_critic_grad_norm"] <= alg.max_grad_norm + 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_ppo_alignment.py -q`

Expected: FAIL because `approx_kl`, `amp_active_count`, `actor_critic_grad_norm` are not returned and AMP update does not call `clip_std`.

### Task 2: 对齐 ParallelismAMPPPO 的 update 核心

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/parallelism_amp_ppo.py`

**Interfaces:**
- Consumes: existing rollout `mu`, `sigma`, `values`, `returns`, `amp_values`, `amp_returns`, `amp_active`。
- Produces: `update() -> dict[str, float]` with `value_loss`, `amp_value_loss`, `surrogate_loss`, `approx_kl`, `actor_critic_grad_norm`, `amp_active_count`, and discriminator metrics。

- [ ] **Step 1: Add old policy statistics to the flattened batch**

Use `storage.mu.reshape(batch_size, -1)` and `storage.sigma.reshape(batch_size, -1)`; index them with the same minibatch permutation as observations/actions. Compute the exact ordinary PPO KL formula under `torch.inference_mode()` and, when `schedule == "adaptive"`, update `self.learning_rate` using the existing `PPO` thresholds (`> 2 * desired_kl` divide by 1.5, `< desired_kl / 2` multiply by 1.5, bounded to `[1e-5, 1e-2]`) and write it to every actor-critic optimizer param group.

- [ ] **Step 2: Make both value losses follow ordinary PPO clipping**

For base value use `target_values_batch` semantics from storage: `value_clipped = old_values_mb + clamp(base_value - old_values_mb, -clip_param, clip_param)`. For AMP use the same expression with `amp_values_mb` as old prediction and `amp_returns_mb` as target. Multiply AMP squared losses by `active_mb` and divide by `active_mb.sum().clamp_min(1.0)`.

- [ ] **Step 3: Keep one shared gradient step and record clipping metrics**

Backpropagate the sum of surrogate, base value, AMP value and entropy losses through the one `actor_critic` optimizer. Capture the return of `nn.utils.clip_grad_norm_` as `actor_critic_grad_norm`, then call `optimizer.step()`. Accumulate finite KL/gradient/count metrics over all minibatches.

- [ ] **Step 4: Restore ordinary PPO post-update std clipping**

After `storage.clear()`, call `self.actor_critic.clip_std(min=self.clip_min_std)` when available. Do not introduce an AMP-specific std or optimizer.

- [ ] **Step 5: Run focused tests**

Run: `env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_ppo_alignment.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_update.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_gae.py -q`

Expected: PASS with finite metrics and inactive rows contributing no AMP value loss.

### Task 3: 完善 AMP 配置和 TensorBoard 记录

**Files:**
- Modify: `Go2Pvcnn/agent/train_cfg.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/parallelism_amp_ppo.py`

**Interfaces:**
- Consumes: existing runner metric dictionary and AMP context。
- Produces: TensorBoard scalars for `amp_active_ratio`, `amp_history_ratio_mean`, `amp_reward_raw_mean`, `amp_reward_gated_mean`, `amp_value_loss`, `discriminator_loss`, `approx_kl`, and `actor_critic_grad_norm`.

- [ ] **Step 1: Add config regression assertions**

Extend `test_parallelism_amp_isolation.py` to assert AMP has the exact shared PPO values, `amp_value_loss_coef == 1.0`, `schedule == "adaptive"`, and that pure/distillation configs remain unchanged.

- [ ] **Step 2: Return metrics from AMP update**

Keep metric keys scalar and finite. Derive active count from `storage.amp_active.bool().sum()` before clear; report replay fill and D metrics without changing D optimizer behavior.

- [ ] **Step 3: Pass metrics through the existing runner logger**

Use the current logger path only for the AMP algorithm branch; do not add AMP keys to pure PPO/distillation branches. Run the focused config and runner tests.

### Task 4: 将 1024 真实 smoke 变成 opt-in 可验收测试

**Files:**
- Modify: `Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py`
- Modify: `Go2Pvcnn/tests/tracking/test_parallelism_amp_training_smoke_probe.py`

**Interfaces:**
- Consumes: AMP launcher and pure PPO checkpoint at `/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/cross_large_complex_ppo/2026-08-26_17-47-24/11d453a/model_19998.pt`。
- Produces: nonzero failure on missing opt-in, bad iteration count, traceback/OOM/NaN/Inf, or missing TensorBoard metrics; zero only after 4 real iterations complete.

- [ ] **Step 1: Add opt-in guard and robust log checks**

Require `RUN_REAL_AMP_1024=1` for `--num-envs 1024 --max-iterations 4`; reject other values in the 1024 test mode. Preserve raw stdout/stderr, match completed iteration numbers, and reject case-insensitive `traceback`, `out of memory`, `nan`, and `inf` (with an allowlist for harmless words if needed).

- [ ] **Step 2: Validate TensorBoard event files**

Locate the run directory emitted by the launcher, read scalar tags using TensorBoard’s `EventAccumulator`, and require `amp_active_ratio`, `amp_history_ratio_mean`, `amp_value_loss`, `discriminator_loss`, `approx_kl`, and `actor_critic_grad_norm` to have at least one scalar. Record collection/update durations and peak CUDA memory in the probe log.

- [ ] **Step 3: Add a pytest contract test without launching Isaac Lab**

Assert the probe contains the opt-in guard, `NUM_ENVS`, `MAX_ITERATIONS`, the supplied checkpoint path/default, and the exact four-iteration set check. Mark the actual launch test `real`/opt-in so default CI remains lightweight.

- [ ] **Step 4: Run the real target test**

Run:

```bash
RUN_REAL_AMP_1024=1 NUM_ENVS=1024 MAX_ITERATIONS=4 \
  /share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py \
  /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/cross_large_complex_ppo/2026-08-26_17-47-24/11d453a/model_19998.pt \
  --num-envs 1024 --max-iterations 4 \
  --log-file /tmp/parallelism_amp_1024_iter4.log
```

Expected: process exit code `0`; four completed iterations; no traceback/OOM/NaN/Inf; required TensorBoard scalar tags present.

### Task 5: 回归、提交与验收记录

**Files:**
- Modify: `notes/log/index.md`
- Create: `notes/log/YYYY-MM-DD-parallelism-amp-1024-real-smoke.md`

- [ ] **Step 1: Run focused unit/config tests**

Run: `env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_*.py Go2Pvcnn/tests/tracking/test_parallelism_amp_*.py -q`

- [ ] **Step 2: Run pure PPO/distillation regressions**

Run: `env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py Go2Pvcnn/tests/tracking/test_parallelism_distillation_*.py -q`

- [ ] **Step 3: Record real output**

Write the exact exit code, completed iterations, TensorBoard tags, peak memory, collection/update timings, GPU, and software versions to the dated log. Do not claim success if the real process was skipped or hardware prevented completion.

- [ ] **Step 4: Commit implementation in small commits**

Use separate commits for algorithm alignment, test/probe changes, and verification log; leave unrelated pre-existing worktree changes untouched.

## Plan Self-Review

- Coverage: dual value/GAE and standstill semantics remain covered by existing tests; this plan adds missing PPO alignment and target-scale smoke coverage.
- No placeholders: all commands, paths, metric names, and failure criteria are explicit.
- Type consistency: `update()` returns scalar metric dict; runner consumes the same keys; storage fields remain `[T, N, 1]` and flattened only inside `update()`.
