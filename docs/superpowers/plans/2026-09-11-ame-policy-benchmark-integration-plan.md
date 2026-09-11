# AME / AME-AMP Policy Benchmark Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `ame` 和 `ame_amp` 接入现有统一 policy benchmark，并用各自最新、训练 iteration 最大的 checkpoint 完成 1024 环境真实 smoke 验证。

**Architecture:** 在 `evaluation/model_adapters.py` 中建立统一的 `BenchmarkRuntime` 契约，保留现有四类模型的默认 adapter，并新增 AME 与 AME-AMP 专用 adapter。`scripts/policy_benchmark.py` 只负责 suite 配置和共享 rollout，模型特有的 cfg、Gym 注册、wrapper、runner、checkpoint 加载及初始观测全部由 adapter 完成。

**Tech Stack:** Python 3.10、PyTorch、Isaac Lab、Gymnasium、RSL-RL、Bash、pytest。

## Global Constraints

- 不修改 AME、AME-AMP 的训练算法、网络结构、reward、planner 或 checkpoint 格式。
- 不改变现有 `amp`、`distillation`、`ppo`、`teacher` 的 benchmark 行为。
- 六类模型共享 `complex_mixed`、`large_runway`、`small_runway` 三套 suite 和相同结果 schema。
- tracking MSE 只统计连续有效的 24 帧窗口；真实 smoke 至少执行 48 transitions。
- AME 必须使用 checkpoint 中保存的 BatchNorm running statistics，推理期间保持 eval mode。
- AME-AMP 只接受 `full_amp_resume`，缺失 AMP value head 或 discriminator 时立即失败。
- batch launcher 顺序运行 suite，任一 suite 失败即停止，并清理上一 suite 的进程组。
- 当前工作区已有相关未提交改动；禁止回滚这些改动，提交时仅暂存本计划或本次明确涉及的评测文件。
- 用户明确要求不使用子 agent，所有步骤在当前会话内执行。

---

### Task 1: 扩展模型 metadata 与 adapter 契约

**Files:**
- Modify: `evaluation/model_adapters.py`
- Modify: `tests/evaluation/test_model_adapters.py`

**Interfaces:**
- Consumes: `ModelSpec`、现有四类 `build_play_mapping()`、AME/AME-AMP 独立 play 入口中的 cfg/wrapper/runner/load 契约。
- Produces: `BenchmarkRuntime(env, base_env, wrapped_env, policy, observations)`、`DefaultBenchmarkAdapter`、`AmeBenchmarkAdapter`、`AmeAmpBenchmarkAdapter`、`build_benchmark_adapter(spec)`。

- [x] **Step 1: 写 metadata 和 adapter 选择的失败测试**

```python
from evaluation.model_adapters import (
    AmeAmpBenchmarkAdapter,
    AmeBenchmarkAdapter,
    DefaultBenchmarkAdapter,
    build_benchmark_adapter,
    resolve_model_spec,
)


def test_resolve_model_spec_maps_ame_adapters(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    ame = resolve_model_spec("ame", checkpoint)
    ame_amp = resolve_model_spec("ame_amp", checkpoint)
    assert ame.experiment == "cross_large_complex_ame"
    assert ame.task_id == "Isaac-Go2-Cross-Large-Complex-PPO-v0"
    assert isinstance(build_benchmark_adapter(ame), AmeBenchmarkAdapter)
    assert ame_amp.experiment == "parallelism_tracking_cross_large_complex_ame_amp"
    assert ame_amp.task_id == "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0"
    assert isinstance(build_benchmark_adapter(ame_amp), AmeAmpBenchmarkAdapter)
    assert isinstance(build_benchmark_adapter(resolve_model_spec("ppo", checkpoint)), DefaultBenchmarkAdapter)


def test_ame_amp_adapter_rejects_non_full_resume():
    with pytest.raises(ValueError, match="complete AME-AMP checkpoint"):
        AmeAmpBenchmarkAdapter.require_full_amp_resume("legacy_policy_warm_start")
```

- [x] **Step 2: 运行测试确认因新类型和类缺失而失败**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_model_adapters.py`

Expected: FAIL，错误指向 `AmeBenchmarkAdapter`、`AmeAmpBenchmarkAdapter` 或 `ame` metadata 尚不存在。

- [x] **Step 3: 实现统一 runtime 与三类 adapter**

```python
@dataclass
class BenchmarkRuntime:
    env: object
    base_env: object
    wrapped_env: object
    policy: object
    observations: object


class DefaultBenchmarkAdapter:
    preserve_planner_owned_reference_cache = False

    def create_env_cfg(self, spec):
        cfg_cls, _ = build_play_mapping()[spec.name]
        return cfg_cls()

    def create_runtime(self, spec, env_cfg, gym_module):
        from agent import get_train_cfg
        from rsl_rl.env import VecEnv
        from rsl_rl.runners import OnPolicyRunner
        from tensordict import TensorDict
        from Go2Pvcnn.scripts.play import _make_env_wrapper

        env = gym_module.make(spec.task_id, cfg=env_cfg, render_mode=None)
        base_env = env.unwrapped
        wrapped = _make_env_wrapper(
            base_env,
            gym_module=gym_module,
            vec_env_cls=VecEnv,
            tensor_dict_cls=TensorDict,
            clip_actions=100.0,
        )
        runner = OnPolicyRunner(wrapped, get_train_cfg(spec.experiment), log_dir=None, device=env_cfg.sim.device)
        runner.load(str(spec.checkpoint), load_optimizer=False)
        return BenchmarkRuntime(env, base_env, wrapped, runner.get_inference_policy(device=wrapped.device), wrapped.consume_initial_observations()[0])


class AmeBenchmarkAdapter:
    preserve_planner_owned_reference_cache = False

    def create_env_cfg(self, spec):
        from ame_baseline.ame_env_cfg import AmeCrossLargeComplexEnvCfg_PLAY
        return AmeCrossLargeComplexEnvCfg_PLAY()

    def create_runtime(self, spec, env_cfg, gym_module):
        import copy
        from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
        from ame_baseline.ame_runner import AmeOnPolicyRunner
        from ame_baseline.ame_train_cfg import get_ame_train_cfg

        env = gym_module.make(spec.task_id, cfg=env_cfg, render_mode=None)
        wrapped = AmeRslRlEnvWrapper(env, clip_actions=100.0)
        runner = AmeOnPolicyRunner(wrapped, copy.deepcopy(get_ame_train_cfg()), log_dir=None, device=env_cfg.sim.device)
        runner.load(str(spec.checkpoint), load_optimizer=False, keep_std=True)
        policy = runner.get_inference_policy(device=wrapped.device)
        observations, _ = wrapped.get_observations()
        return BenchmarkRuntime(env, env.unwrapped, wrapped, policy, observations)


class AmeAmpBenchmarkAdapter(AmeBenchmarkAdapter):
    preserve_planner_owned_reference_cache = True

    @staticmethod
    def require_full_amp_resume(mode):
        if mode != "full_amp_resume":
            raise ValueError("benchmark requires a complete AME-AMP checkpoint")

    def create_env_cfg(self, spec):
        from ame_baseline.ame_amp_env_cfg import AmeParallelismAmpCrossLargeComplexEnvCfg
        return AmeParallelismAmpCrossLargeComplexEnvCfg()
```

`AmeAmpBenchmarkAdapter.create_runtime()` 还必须按独立 play 入口幂等注册 `AME_AMP_ENV_ID`，调用 `attach_trajectory_manager_if_enabled()`，使用 `AmeRslRlEnvWrapper`、`AmeAmpOnPolicyRunner` 和 `get_ame_amp_train_cfg()`，并把 `load_amp_checkpoint(..., keep_std=True)` 返回值交给 `require_full_amp_resume()`。

- [x] **Step 4: 运行 focused tests 确认通过**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_model_adapters.py`

Expected: PASS，至少覆盖六类 metadata、三类 adapter 选择和 AME-AMP mode 拒绝逻辑。

### Task 2: 让共享 rollout 使用 adapter runtime

**Files:**
- Modify: `scripts/policy_benchmark.py`
- Modify: `tests/evaluation/test_policy_benchmark_static.py`

**Interfaces:**
- Consumes: `build_benchmark_adapter(spec)` 和 `BenchmarkRuntime`。
- Produces: 六类 CLI 选择；模型初始化后统一获得 `env`、`base_env`、`wrapped_env`、`policy`、`obs`。

- [x] **Step 1: 写 CLI 和调用边界的失败测试**

```python
def test_policy_benchmark_supports_ame_adapters():
    source = (ROOT / "scripts/policy_benchmark.py").read_text(encoding="utf-8")
    assert '"ame"' in source
    assert '"ame_amp"' in source
    assert "build_benchmark_adapter" in source
    assert "adapter.create_env_cfg" in source
    assert "adapter.create_runtime" in source
```

- [x] **Step 2: 运行测试确认共享脚本尚未使用 adapter**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_policy_benchmark_static.py::test_policy_benchmark_supports_ame_adapters`

Expected: FAIL，缺少新 choices 或 adapter 调用。

- [x] **Step 3: 修改 `_run_rollout()` 的模型初始化段**

```python
spec = resolve_model_spec(args.experiment_type, args.checkpoint)
adapter = build_benchmark_adapter(spec)
env_cfg = adapter.create_env_cfg(spec)
env_cfg.scene.num_envs = int(args.num_envs)
env_cfg.sim.device = args.device
env_cfg = build_suite_config(args.suite, env_cfg, conditions[0])
_enable_benchmark_contact_sensors(env_cfg)
if spec.requires_reference_manager and not adapter.preserve_planner_owned_reference_cache:
    env_cfg.planner_owned_reference_cache = False
runtime = adapter.create_runtime(spec, env_cfg, gym)
env = runtime.env
base_env = runtime.base_env
wrapped_env = runtime.wrapped_env
policy = runtime.policy
obs = runtime.observations
```

保留后续 `_install_pre_reset_collision_capture()`、`get_parallelism_reference_manager()`、24 帧 `ValidWindowBuffer`、碰撞/跌倒/位移累计与 `ResultWriter` 路径不变。删除脚本中不再需要的通用 runner/wrapper imports。

- [x] **Step 4: 运行静态和 model adapter tests**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_model_adapters.py tests/evaluation/test_policy_benchmark_static.py`

Expected: PASS。

### Task 3: 扩展 single、batch 和 smoke shell/Python 入口

**Files:**
- Modify: `scripts/run_policy_benchmark.sh`
- Modify: `scripts/run_policy_benchmark_batch.sh`
- Modify: `tests/evaluation/run_policy_benchmark_1024_smoke.py`
- Modify: `tests/evaluation/test_policy_benchmark_static.py`

**Interfaces:**
- Consumes: `--experiment-type ame|ame_amp` 和显式 checkpoint 文件。
- Produces: 自动目录 `eval_output/<timestamp>_<experiment-type>_<weight>_<sha256>/`、对应 RL task 的 `run_index.jsonl` 记录、单 suite smoke 命令。

- [x] **Step 1: 写 launcher 允许列表和 task 映射的失败测试**

```python
def test_launchers_support_ame_and_ame_amp():
    single = (ROOT / "scripts/run_policy_benchmark.sh").read_text(encoding="utf-8")
    batch = (ROOT / "scripts/run_policy_benchmark_batch.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "tests/evaluation/run_policy_benchmark_1024_smoke.py").read_text(encoding="utf-8")
    for model in ("ame", "ame_amp"):
        assert model in single
        assert model in batch
        assert model in smoke
    assert "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0" in single
```

- [x] **Step 2: 运行测试确认 launcher 尚不接受新类型**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_policy_benchmark_static.py::test_launchers_support_ame_and_ame_amp`

Expected: FAIL。

- [x] **Step 3: 扩展三处允许列表和 single launcher task case**

```bash
case "$experiment_type" in
  amp) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AMP-v0" ;;
  distillation) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-Distillation-v0" ;;
  ppo) task_id="Isaac-Go2-Cross-Large-Complex-PPO-v0" ;;
  teacher) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-v0" ;;
  ame) task_id="Isaac-Go2-Cross-Large-Complex-PPO-v0" ;;
  ame_amp) task_id="Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0" ;;
  *) echo "unknown experiment type: $experiment_type" >&2; exit 2 ;;
esac
```

batch regex 和 smoke argparse choices 同步扩展为六类；不加入 `--smoke-test` 或 `--max-steps` 默认值，保持 batch 默认全量三 suite。

- [x] **Step 4: 运行 shell syntax 和 focused tests**

Run: `bash -n scripts/run_policy_benchmark.sh scripts/run_policy_benchmark_batch.sh`

Expected: exit 0。

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_policy_benchmark_static.py`

Expected: PASS。

### Task 4: 完整回归与 checkpoint 预检

**Files:**
- No source changes.

**Interfaces:**
- Consumes: 两个固定 checkpoint 和所有 evaluation/AME tests。
- Produces: checkpoint iteration/signature/head/discriminator/finite 证据与回归测试结果。

- [x] **Step 1: 检查 AME checkpoint**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -c 'import torch; p="logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt"; c=torch.load(p,map_location="cpu"); s=c["model_state_dict"]; assert c["iter"]==19999; assert c["ame_architecture_signature"]=="ame_xyz_semantic_v1_c6_s16_mha64_h16"; assert all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v)); print(c["iter"], len(s))'
```

Expected: 第一列打印 `19999`，第二列打印实际模型 state 字典项数，并 exit 0。

- [x] **Step 2: 检查 AME-AMP checkpoint**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -c 'import torch; p="logs/rsl_rl/parallelism_tracking_cross_large_complex_ame_amp/2026-09-10_15-42-10/f234425/model_3499.pt"; c=torch.load(p,map_location="cpu"); s=c["model_state_dict"]; d=c["amp_discriminator_state_dict"]; assert c["iter"]==3499; assert any(k.startswith("amp_value_head.") for k in s); assert all(torch.isfinite(v).all() for q in (s,d) for v in q.values() if torch.is_tensor(v)); print(c["iter"], len(s), len(d))'
```

Expected: 第一列打印 `3499`，后两列分别打印实际模型和 discriminator state 字典项数，并 exit 0。

- [x] **Step 3: 运行完整相关测试**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation Go2Pvcnn/tests/ame_baseline
```

Expected: 0 failures。

### Task 5: 运行两个 1024 环境真实 smoke 并验收产物

**Files:**
- Create runtime output under: `eval_output/ame_integration_smoke_20260911/`

**Interfaces:**
- Consumes: 固定 AME/AME-AMP checkpoint、`complex_mixed` suite、1024 env、48 transitions。
- Produces: 每个模型的 episode shards、`progress.json`、`run_manifest.json`、`summary.json` 和无残留进程证据。

- [x] **Step 1: 运行 AME smoke**

Run:

```bash
export PATH=/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin:$PATH
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python tests/evaluation/run_policy_benchmark_1024_smoke.py --experiment-type ame --checkpoint logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt --num-envs 1024 --transitions 48 --output-dir eval_output/ame_integration_smoke_20260911/ame --device cuda:0
```

Expected: exit 0，`valid_windows >= 1`，日志无 traceback/OOM/NaN/Inf。

- [x] **Step 2: 清理并确认 AME 进程退出后运行 AME-AMP smoke**

Run:

```bash
export PATH=/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin:$PATH
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python tests/evaluation/run_policy_benchmark_1024_smoke.py --experiment-type ame_amp --checkpoint logs/rsl_rl/parallelism_tracking_cross_large_complex_ame_amp/2026-09-10_15-42-10/f234425/model_3499.pt --num-envs 1024 --transitions 48 --output-dir eval_output/ame_integration_smoke_20260911/ame_amp --device cuda:0
```

Expected: exit 0，checkpoint mode 为 `full_amp_resume`，`valid_windows >= 1`，日志无 traceback/OOM/NaN/Inf。

- [x] **Step 3: 验证同 schema 结果和数值有限**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -c 'import json,math,pathlib; roots=[pathlib.Path("eval_output/ame_integration_smoke_20260911/ame/ame/complex_mixed"),pathlib.Path("eval_output/ame_integration_smoke_20260911/ame_amp/ame_amp/complex_mixed")]; required=("summary.json","progress.json","run_manifest.json"); [(r/f).is_file() or (_ for _ in ()).throw(AssertionError(r/f)) for r in roots for f in required]; summaries=[json.loads((r/"summary.json").read_text()) for r in roots]; assert all(s["valid_windows"]>=1 and math.isfinite(s["env_steps_per_second"]) and math.isfinite(s["elapsed_s"]) for s in summaries); print([(s["episodes"],s["valid_windows"],s["env_steps_per_second"]) for s in summaries])'
```

Expected: 打印两个 `(episodes, valid_windows, env_steps_per_second)` 元组并 exit 0。

- [x] **Step 4: 检查无 benchmark Isaac/Kit 残留进程**

Run: `pgrep -af 'policy_benchmark.py|run_policy_benchmark|isaac-sim|kit'`

Expected: 不出现本次两个 smoke 创建的进程。

### Task 6: 最终 diff、提交与正式命令交付

**Files:**
- Modify: `docs/superpowers/plans/2026-09-11-ame-policy-benchmark-integration-plan.md`（勾选实际完成步骤）

**Interfaces:**
- Consumes: 所有测试和 smoke 结果。
- Produces: 一个范围明确的 Git 提交以及两条不带 `--smoke-test` 的完整三 suite 命令。

- [x] **Step 1: 检查 diff 与格式**

Run: `git diff --check`

Expected: exit 0。

Run: `git diff -- evaluation/model_adapters.py scripts/policy_benchmark.py scripts/run_policy_benchmark.sh scripts/run_policy_benchmark_batch.sh tests/evaluation/test_model_adapters.py tests/evaluation/test_policy_benchmark_static.py tests/evaluation/run_policy_benchmark_1024_smoke.py docs/superpowers/plans/2026-09-11-ame-policy-benchmark-integration-plan.md`

Expected: 只包含本设计范围内变更，并保留这些文件中已存在的相关 benchmark 修改。

- [x] **Step 2: 只暂存明确文件并提交**

```bash
git add evaluation/model_adapters.py scripts/policy_benchmark.py scripts/run_policy_benchmark.sh scripts/run_policy_benchmark_batch.sh tests/evaluation/test_model_adapters.py tests/evaluation/test_policy_benchmark_static.py tests/evaluation/run_policy_benchmark_1024_smoke.py docs/superpowers/plans/2026-09-11-ame-policy-benchmark-integration-plan.md
git diff --cached --check
git commit -m "feat: benchmark AME policy variants"
```

- [x] **Step 3: 交付 AME 正式 batch 命令**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn && \
export PATH=/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin:$PATH && \
scripts/run_policy_benchmark_batch.sh \
  --experiment-type ame \
  --checkpoint logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt \
  --num-envs 1024 \
  --device cuda:0 \
  --headless
```

- [x] **Step 4: 交付 AME-AMP 正式 batch 命令**

```bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn && \
export PATH=/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin:$PATH && \
scripts/run_policy_benchmark_batch.sh \
  --experiment-type ame_amp \
  --checkpoint logs/rsl_rl/parallelism_tracking_cross_large_complex_ame_amp/2026-09-10_15-42-10/f234425/model_3499.pt \
  --num-envs 1024 \
  --device cuda:0 \
  --headless
```
