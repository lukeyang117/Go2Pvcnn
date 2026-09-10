# Go2 AME Parallelism AMP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改既有 AME、Parallelism AMP、PPO 或 Distillation 文件的前提下，从 AME `model_19999.pt` warm-start，新增保留 AME CNN/MHA/BatchNorm 的 Parallelism AMP 训练入口，完成 3500 iteration 续训并通过 1024 环境真实 smoke test。

**Architecture:** 新增 `AmpActorCriticAME` 继承 AME 网络，只增加 detached AMP value head；新增 AME-AMP 环境配置继承 AME 的六通道地图和 mixed-terrain scene，运行时复用 `ParallelismAmpEnv` 的 24 帧 planner ring。新增 runner 通过 `OnPolicyRunner` 的现有 AMP 算法路径注入网络，严格区分 AME legacy warm-start 与完整 AME-AMP resume，并为新实验建立独立输出目录。

**Tech Stack:** Python 3.10、PyTorch、Isaac Lab、Gymnasium、RSL-RL `ParallelismAMPPPO`、pytest、Bash。

## Global Constraints

- AMP expert 使用 Parallelism planner 的 24 帧、39 维 state window；不使用 AME 轨迹或外部动作数据集。
- Actor/Critic 输入保持 AME 的六通道 `[x_local,y_local,z_local,terrain,small,large]` 16x16 map 和 AME state groups。
- `amp_value_head` 的 AME critic feature 必须 `detach`，AMP value loss 不得更新 AME encoder、MHA、base Critic 或 Actor。
- AMP actor reward weight 为 `0..499=0`、`500=0`、`550=0.05`、`600+=0.1`；AMP critic 与 D 在 warm-up 期间继续训练。
- PPO 使用 `num_steps_per_env=40`、5 epochs、4 mini-batches、lr `1e-3`、adaptive-KL `desired_kl=0.01`、`max_grad_norm=1.0`。
- AMP 使用 window 24、`amp_reward_weight=0.1`、`amp_value_loss_coef=1.0`、D lr `1e-4`、D epochs 2、D batch 4096、replay 32768。
- planner 无解、standstill、reset 清空对应 AMP ring；重新连续累计满 24 帧前 `amp_active=0`，base PPO 仍训练。
- rollout 40 步与 AMP ring 24 帧独立；ring 跨 PPO iteration 保留，重规划 transition 必须按 `B0 -> B1` 对齐。
- warm-start 源为 `/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt`，不加载源 PPO optimizer；std、39 个 AME 模型 tensor 必须保留。
- 新实验输出为 `logs/rsl_rl/parallelism_tracking_cross_large_complex_ame_amp/<timestamp>/<git-hash>/`，3500 次首次训练最终保存 `model_3499.pt`。
- 所有新增连接代码放在新文件；不得编辑既有 AME、PPO、AMP、Distillation、Teacher、Gym 全局注册文件。

---

### Task 1: Add AME AMP actor-critic and isolated unit tests

**Files:**
- Create: `Go2Pvcnn/ame_baseline/amp_actor_critic_ame.py`
- Create: `Go2Pvcnn/tests/ame_baseline/test_amp_actor_critic_ame.py`

**Interfaces:**
- Consumes: `ame_baseline.actor_critic_ame.ActorCriticAME` and policy dimensions `(num_actor_obs, num_critic_obs, num_actions)`.
- Produces: `AmpActorCriticAME.evaluate_amp(critic_observations, amp_active, history_ratio) -> Tensor[N,1]`, `load_common_state_dict(state_dict)`, and `_last_critic_feature` cache used by the new runner.

- [ ] **Step 1: Write failing tests for dimensions, zero AMP head, state loading, and gradient isolation.**

```python
def test_amp_head_and_context_shape():
    model = AmpActorCriticAME(1581, 1584, 12, map_channels=6, map_size=16, mha_dim=64, num_heads=16)
    critic = torch.randn(8, 1584)
    value = model.evaluate_amp(critic, torch.ones(8), torch.ones(8))
    assert value.shape == (8, 1)
    assert torch.equal(value, torch.zeros_like(value))

def test_amp_value_loss_does_not_update_ame_encoder():
    model = AmpActorCriticAME(1581, 1584, 12, map_channels=6, map_size=16, mha_dim=64, num_heads=16)
    before = {name: value.detach().clone() for name, value in model.map_cnn.named_parameters()}
    value = model.evaluate_amp(torch.randn(8, 1584), torch.ones(8), torch.ones(8))
    value.square().mean().backward()
    assert all(param.grad is None for param in model.map_cnn.parameters())
    assert any(param.grad is not None for param in model.amp_value_head.parameters())
    assert all(torch.equal(before[name], value.detach()) for name, value in model.map_cnn.named_parameters())

def test_legacy_ame_state_is_loaded_without_amp_keys():
    source = ActorCriticAME(1581, 1584, 12, map_channels=6, map_size=16, mha_dim=64, num_heads=16)
    target = AmpActorCriticAME(1581, 1584, 12, map_channels=6, map_size=16, mha_dim=64, num_heads=16)
    target.load_common_state_dict(source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the new class does not exist.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_amp_actor_critic_ame.py -q`

Expected: collection failure mentioning `amp_actor_critic_ame` is missing.

- [ ] **Step 3: Implement the new network.**

Implement `AmpActorCriticAME(ActorCriticAME)` with:

```python
class AmpActorCriticAME(ActorCriticAME):
    def __init__(self, *args, amp_value_hidden_dims=(256, 128), **kwargs): ...
    def evaluate(self, critic_observations, **kwargs): ...
    def evaluate_amp(self, critic_observations, amp_active, history_ratio): ...
    def load_common_state_dict(self, state_dict): ...
```

`evaluate()` must call the AME critic encoder once, store a detached/non-detached feature pair for the immediately following `evaluate_amp()`, and return base value. `evaluate_amp()` must reuse the cached feature when observation identity and batch shape match; otherwise call the encoder once, immediately detach it, concatenate two context columns, and run `Linear(114,256)-ELU-Linear(256,128)-ELU-Linear(128,1)`. Initialize all AMP linear layers orthogonally and zero the final layer. `load_common_state_dict()` must require every non-AMP key and exact shape, load `std` and BatchNorm buffers, and reject missing or mismatched legacy keys.

- [ ] **Step 4: Run the focused tests and check finite outputs and gradient boundaries.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_amp_actor_critic_ame.py -q`

Expected: all tests pass; AMP output is exactly zero before training, encoder grads are absent for AMP-only loss, and legacy AME tensors match bit-for-bit.

### Task 2: Add isolated AME-AMP environment and training configuration

**Files:**
- Create: `Go2Pvcnn/ame_baseline/ame_amp_env_cfg.py`
- Create: `Go2Pvcnn/ame_baseline/ame_amp_train_cfg.py`
- Create: `Go2Pvcnn/tests/ame_baseline/test_ame_amp_config.py`

**Interfaces:**
- Consumes: `AmeCrossLargeComplexEnvCfg`, `ParallelismAmpEnv`, and existing AME/AMP configuration values.
- Produces: `AmeParallelismAmpCrossLargeComplexEnvCfg`, `get_ame_amp_train_cfg()`, and runtime metadata constants (`AME_AMP_EXPERIMENT_NAME`, `AME_AMP_ENV_ID`).

- [ ] **Step 1: Write config tests before implementation.**

```python
def test_config_preserves_ame_observations_and_amp_fields():
    cfg = AmeParallelismAmpCrossLargeComplexEnvCfg()
    assert cfg.experiment_name == "parallelism_tracking_cross_large_complex_ame_amp"
    assert cfg.planner_owned_reference_cache is True
    assert cfg.parallelism_plan_batch_size == 1024
    assert cfg.amp_window_frames == 24 and cfg.amp_dt == 0.02
    train = get_ame_amp_train_cfg()
    assert train["policy"]["class_name"] == "AmpActorCriticAME"
    assert train["algorithm"]["class_name"] == "ParallelismAMPPPO"
    assert train["num_steps_per_env"] == 40
    assert train["algorithm"]["schedule"] == "adaptive"

def test_amp_config_does_not_mutate_ame_config():
    assert get_ame_train_cfg()["policy"]["class_name"] == "ActorCriticAME"
```

- [ ] **Step 2: Run the config tests and confirm missing-module failure.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_ame_amp_config.py -q`

Expected: collection failure until the new config modules exist.

- [ ] **Step 3: Implement the configuration modules.**

`AmeParallelismAmpCrossLargeComplexEnvCfg` inherits `AmeCrossLargeComplexEnvCfg`, sets the independent experiment name, `planner_owned_reference_cache=True`, `parallelism_plan_batch_size=1024`, `amp_window_frames=24`, and `amp_dt=0.02`. `get_ame_amp_train_cfg()` returns a deep-copy-safe dictionary with AME map settings (`map_channels=6`, `map_size=16`, `mha_dim=64`, `num_heads=16`, hidden dims `[512,256,128]`), ordinary PPO fields copied exactly from AME, and the AMP-only fields listed in Global Constraints. Do not import or mutate `agent.train_cfg`.

- [ ] **Step 4: Run config tests and compare all shared PPO values against AME.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_ame_amp_config.py -q`

Expected: PASS with unchanged AME config and exact shared PPO values.

### Task 3: Add AME-AMP runner, checkpoint lifecycle, and tests

**Files:**
- Create: `Go2Pvcnn/ame_baseline/ame_amp_runner.py`
- Create: `Go2Pvcnn/tests/ame_baseline/test_ame_amp_checkpoint.py`

**Interfaces:**
- Consumes: `OnPolicyRunner`, `ParallelismAMPPPO`, `AmpActorCriticAME`, AME source checkpoint, and `torch.sha256` file metadata.
- Produces: `AmeAmpOnPolicyRunner.load_amp_checkpoint(path, keep_std=True)`, `AmeAmpOnPolicyRunner.save(path, infos=None)`, and checkpoint metadata keys `ame_amp_signature`, `ame_source_checkpoint`, `ame_source_sha256`, `ame_source_iteration`.

- [ ] **Step 1: Write checkpoint tests for legacy warm-start, full resume, optimizer restoration, and rejection.**

Use a small CPU mock VecEnv exposing observations `[2,1581]`, critic observations `[2,1584]`, and 12 actions. Save a source AME checkpoint containing 39 model tensors and `iter=19999`, then assert:

```python
runner.load_amp_checkpoint(source_path)
assert runner.current_learning_iteration == 0
assert torch.equal(runner.alg.actor_critic.std, source_state["std"])
assert not any(key.startswith("amp_value_head.") for key in source_state)

runner.save(full_path)
resumed.load_amp_checkpoint(full_path)
assert resumed.current_learning_iteration == runner.current_learning_iteration
assert resumed.alg.amp_discriminator.state_dict().keys() == runner.alg.amp_discriminator.state_dict().keys()
assert resumed.alg.optimizer.state_dict()["state"]
```

Also assert a checkpoint containing only `amp_value_head.*` or missing AME signature raises `IncompleteAMPCheckpointError`/`ValueError` before training.

- [ ] **Step 2: Run checkpoint tests and confirm they fail before runner implementation.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_ame_amp_checkpoint.py -q`

Expected: missing `AmeAmpOnPolicyRunner` failure.

- [ ] **Step 3: Implement strict warm-start and full-resume behavior.**

Before `OnPolicyRunner.__init__`, inject only the new class into `rsl_rl.runners.on_policy_runner.AmpActorCriticAME` so its existing `eval(self.policy_cfg.pop("class_name"))` path resolves without editing RSL-RL. `load_amp_checkpoint()` must:

1. load CPU checkpoint and validate `model_state_dict`, AME signature/dimensions, and finite tensors;
2. detect legacy versus complete AMP state by the paired presence of `amp_value_head.*` and `amp_discriminator_state_dict`;
3. for legacy AME, call `load_common_state_dict`, keep source std, reset AMP D/head and both new optimizers, set local iteration 0, and record source absolute path/SHA256/iteration;
4. for complete AME-AMP, strictly load actor-critic, AMP discriminator, actor optimizer, discriminator optimizer, and checkpoint iteration;
5. reject partial AMP checkpoints and incompatible dimensions.

Override `save()` by calling the existing runner save first, then append metadata without dropping the D state, D optimizer state, actor optimizer state, or `iter`. Preserve source metadata across resumes.

- [ ] **Step 4: Run checkpoint tests and verify source checkpoint values.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_ame_amp_checkpoint.py -q`

Expected: source `iter=19999`, 39 AME tensors and std range `0.2952693..0.3883149` are preserved; full resume restores both optimizer states.

### Task 4: Add runtime training/play entrypoints and static tests

**Files:**
- Create: `Go2Pvcnn/scripts/train_cross_large_complex_ame_amp.py`
- Create: `Go2Pvcnn/scripts/train_cross_large_complex_ame_amp_headless.sh`
- Create: `Go2Pvcnn/scripts/play_cross_large_complex_ame_amp.py`
- Create: `Go2Pvcnn/tests/ame_baseline/test_ame_amp_entrypoints.py`

**Interfaces:**
- Consumes: Task 2 config, Task 3 runner, existing runtime trajectory manager factory, and `AmeRslRlEnvWrapper`.
- Produces: a new Gym registration local to the process, `MAX_ITERATIONS=3500`/`NUM_ENVS=1024` defaults, isolated log directory, and deterministic play loader.

- [ ] **Step 1: Write static entrypoint tests.**

```python
def test_shell_has_requested_warm_start_and_defaults():
    text = Path("Go2Pvcnn/scripts/train_cross_large_complex_ame_amp_headless.sh").read_text()
    assert "model_19999.pt" in text
    assert "MAX_ITERATIONS:-3500" in text and "NUM_ENVS:-1024" in text
    assert "parallelism_tracking_cross_large_complex_ame_amp" in text

def test_train_script_registers_only_runtime_experiment():
    text = Path("Go2Pvcnn/scripts/train_cross_large_complex_ame_amp.py").read_text()
    assert "gym.register" in text and "AmeParallelismAmpCrossLargeComplexEnvCfg" in text
    assert "AmeAmpOnPolicyRunner" in text
```

- [ ] **Step 2: Implement the training entrypoint.**

The script must launch Isaac Lab before importing runtime-only modules, register `AME_AMP_ENV_ID` with entry point `tracking.amp_env:ParallelismAmpEnv` and the new config, import `go2_pvcnn.tasks.register_envs` for scene dependencies, attach the existing trajectory manager factory to the unwrapped environment, wrap with `AmeRslRlEnvWrapper`, create the isolated timestamp/hash log directory, dump configs, construct `AmeAmpOnPolicyRunner`, load the default AME source through `load_amp_checkpoint()`, and call `runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)`. The `finally` block must close env and Isaac app.

- [ ] **Step 3: Implement the headless shell and play entrypoint.**

The shell must `cd` to the Go2Pvcnn repository, set Isaac Python/CUDA/Kit variables, resolve `CHECKPOINT` from the fixed AME source unless overridden, pass `--resume --checkpoint --keep_std`, default to 1024 environments and 3500 iterations, and use `exec` so signals reach Isaac. Play must register the same runtime environment, load only a complete AME-AMP checkpoint, set eval/no-grad mode, and close the environment/app in `finally`.

- [ ] **Step 4: Run static tests and shell syntax checks.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_ame_amp_entrypoints.py -q && bash -n Go2Pvcnn/scripts/train_cross_large_complex_ame_amp_headless.sh`

Expected: PASS and shell exit code 0.

### Task 5: Add real Isaac Lab smoke test and integrate test commands

**Files:**
- Create: `Go2Pvcnn/tests/ame_baseline/run_ame_amp_isaac_smoke.sh`
- Create: `Go2Pvcnn/tests/ame_baseline/test_ame_amp_isaac_smoke_static.py`

**Interfaces:**
- Consumes: Task 4 headless training entrypoint and fixed AME source checkpoint.
- Produces: real `1024 env x 4 iteration x 40 rollout steps` run, a temporary output directory containing `model_3.pt`, finite checkpoint tensors, and no process owned by the smoke command after exit.

- [ ] **Step 1: Add static smoke command assertions.**

Assert the smoke script invokes `MAX_ITERATIONS=4`, `NUM_ENVS=1024`, the AME-AMP shell, and uses a temporary unique output root or explicit `OUTPUT_DIR` so it cannot overwrite a production run.

- [ ] **Step 2: Implement the smoke runner.**

Use `set -euo pipefail`, trap cleanup, launch headless training with `MAX_ITERATIONS=4 NUM_ENVS=1024`, wait for normal exit, locate the generated run directory, assert `model_3.pt` exists, load it with the Isaac Python interpreter, assert every tensor is finite, and return the training process exit code. Cleanup must run for both success and failure.

- [ ] **Step 3: Run static smoke checks.**

Run: `pytest Go2Pvcnn/tests/ame_baseline/test_ame_amp_isaac_smoke_static.py -q && bash -n Go2Pvcnn/tests/ame_baseline/run_ame_amp_isaac_smoke.sh`

Expected: PASS and shell exit code 0.

- [ ] **Step 4: Run the real Isaac smoke test.**

Run: `bash Go2Pvcnn/tests/ame_baseline/run_ame_amp_isaac_smoke.sh`

Expected: Isaac exits with code 0 after four iterations; output contains policy/critic dimensions `1581/1584`, AMP discriminator initialization, `model_3.pt`, all checkpoint tensors finite, and no residual process from this entrypoint.

### Task 6: Final regression, review, and commit on the current branch

**Files:**
- Modify: none of the existing implementation files; only new files from Tasks 1-5.

- [ ] **Step 1: Run the focused AME-AMP suite.**

Run: `pytest Go2Pvcnn/tests/ame_baseline -q`

Expected: all new unit/static tests pass.

- [ ] **Step 2: Run existing AME and AMP regressions.**

Run: `pytest Go2Pvcnn/tests/test_ame_observations.py Go2Pvcnn/tests/tracking/test_parallelism_amp_history.py Go2Pvcnn/tests/tracking/test_parallelism_amp_standstill.py Go2Pvcnn/tests/tracking/test_parallelism_amp_state_encoding.py Go2Pvcnn/tests/tracking/test_parallelism_amp_time_alignment.py -q`

Expected: existing behavior remains green.

- [ ] **Step 3: Verify scope before commit.**

Run: `git diff --check && git status --short`

Confirm only the new plan, AME-AMP implementation files, and tests are staged; do not stage or revert unrelated user changes.

- [ ] **Step 4: Commit the plan and implementation on branch `parallelism-amp`.**

```bash
git add docs/superpowers/plans/2026-09-10-go2-ame-parallelism-amp-plan.md \
  Go2Pvcnn/ame_baseline/amp_actor_critic_ame.py \
  Go2Pvcnn/ame_baseline/ame_amp_env_cfg.py \
  Go2Pvcnn/ame_baseline/ame_amp_train_cfg.py \
  Go2Pvcnn/ame_baseline/ame_amp_runner.py \
  Go2Pvcnn/scripts/train_cross_large_complex_ame_amp.py \
  Go2Pvcnn/scripts/train_cross_large_complex_ame_amp_headless.sh \
  Go2Pvcnn/scripts/play_cross_large_complex_ame_amp.py \
  Go2Pvcnn/tests/ame_baseline
git commit -m "feat: add AME Parallelism AMP warm start"
```

Expected: commit is created on the current `parallelism-amp` branch and existing dirty files remain unstaged and untouched.
