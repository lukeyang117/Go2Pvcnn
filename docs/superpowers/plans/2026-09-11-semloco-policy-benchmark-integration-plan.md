# SemLoco Policy Benchmark Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `semloco` 接入现有统一 policy benchmark，并用最新 `model_19999.pt` 完成 1024 环境真实 smoke 验证。

**Architecture:** SemLoco 通过 `evaluation/model_adapters.py` 中的 `DefaultBenchmarkAdapter` 复用标准 `_make_env_wrapper`、`OnPolicyRunner` 和已有共享 rollout。只扩展模型 metadata、PLAY cfg 映射和命令入口白名单；Shadow Parallelism 仍只提供 valid-only 24 帧 MSE，不改变 SemLoco observation 或 action。

**Tech Stack:** Python 3.10、PyTorch、Isaac Lab、Gymnasium、RSL-RL、Bash、pytest。

## Global Constraints

- 不修改 SemLoco 的训练算法、网络结构、reward、planner、checkpoint 格式或现有 play 入口。
- SemLoco 使用 `cross_large_complex_semloco`、`Isaac-Go2-Cross-Large-Complex-SemLoco-v0`、`SemlocoCrossLargeComplexEnvCfg_PLAY` 和 `deployment` observation。
- SemLoco 必须复用 `DefaultBenchmarkAdapter`，不增加无行为差异的专用 adapter。
- 三套正式 suite 固定为 `complex_mixed`、`large_runway`、`small_runway`，共用现有结果 schema 和自动输出目录。
- tracking MSE 只统计连续有效的 24 帧窗口；真实 smoke 固定为 1024 env、48 transitions、`complex_mixed`。
- batch launcher 保持顺序运行、失败即停和进程组清理行为。
- 当前工作区有用户未提交改动；不得回滚，提交时只暂存本任务文件。
- 用户明确要求不使用子 agent，计划在当前会话内通过 `superpowers:executing-plans` 执行。

---

### Task 1: 注册 SemLoco metadata 与默认 adapter

**Files:**
- Modify: `tests/evaluation/test_model_adapters.py`
- Modify: `evaluation/model_adapters.py`

**Interfaces:**
- Consumes: `resolve_model_spec(kind: str, checkpoint: str | Path) -> ModelSpec`、`build_benchmark_adapter(spec)`、`build_play_mapping()`。
- Produces: `semloco` 的 `ModelSpec` 以及由 `DefaultBenchmarkAdapter` 创建的标准 SemLoco runtime。

- [ ] **Step 1: 写 SemLoco metadata 和 adapter 选择的失败测试**

在 `tests/evaluation/test_model_adapters.py` 增加：

```python
def test_resolve_model_spec_maps_semloco_to_default_adapter(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    semloco = resolve_model_spec("semloco", checkpoint)

    assert semloco.experiment == "cross_large_complex_semloco"
    assert semloco.task_id == "Isaac-Go2-Cross-Large-Complex-SemLoco-v0"
    assert semloco.observation_kind == "deployment"
    assert semloco.requires_reference_manager is True
    assert semloco.adapter_kind == "default"
    assert isinstance(build_benchmark_adapter(semloco), DefaultBenchmarkAdapter)
```

- [ ] **Step 2: 运行测试确认因 SemLoco metadata 缺失而失败**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_model_adapters.py::test_resolve_model_spec_maps_semloco_to_default_adapter
```

Expected: FAIL，错误为 `unknown experiment type: semloco`。

- [ ] **Step 3: 增加 metadata 和 PLAY cfg 映射**

在 `_MODEL_TABLE` 增加：

```python
"semloco": (
    "cross_large_complex_semloco",
    "Isaac-Go2-Cross-Large-Complex-SemLoco-v0",
    "deployment",
    True,
    "default",
),
```

在 `build_play_mapping()` 懒加载并映射：

```python
from semloco.semloco_env_cfg import SemlocoCrossLargeComplexEnvCfg_PLAY

"semloco": (SemlocoCrossLargeComplexEnvCfg_PLAY, _MODEL_TABLE["semloco"][1]),
```

- [ ] **Step 4: 运行 model adapter tests 确认通过**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_model_adapters.py
```

Expected: PASS。

### Task 2: 扩展 Python、single、batch 和 smoke 入口

**Files:**
- Modify: `tests/evaluation/test_policy_benchmark_static.py`
- Modify: `scripts/policy_benchmark.py`
- Modify: `scripts/run_policy_benchmark.sh`
- Modify: `scripts/run_policy_benchmark_batch.sh`
- Modify: `tests/evaluation/run_policy_benchmark_1024_smoke.py`

**Interfaces:**
- Consumes: `--experiment-type semloco` 和显式 checkpoint 文件。
- Produces: single-suite 执行、三-suite batch 执行和 1024-env smoke helper；batch 输出继续位于 `eval_output/<timestamp>_semloco_<weight>_<hash>/`。

- [ ] **Step 1: 写四个入口都接受 SemLoco 的失败测试**

在 `tests/evaluation/test_policy_benchmark_static.py` 增加：

```python
def test_launchers_support_semloco() -> None:
    benchmark = (ROOT / "scripts/policy_benchmark.py").read_text(encoding="utf-8")
    single = (ROOT / "scripts/run_policy_benchmark.sh").read_text(encoding="utf-8")
    batch = (ROOT / "scripts/run_policy_benchmark_batch.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "tests/evaluation/run_policy_benchmark_1024_smoke.py").read_text(encoding="utf-8")

    for source in (benchmark, single, batch, smoke):
        assert "semloco" in source
    assert "Isaac-Go2-Cross-Large-Complex-SemLoco-v0" in single
```

- [ ] **Step 2: 运行测试确认入口白名单缺失而失败**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_policy_benchmark_static.py::test_launchers_support_semloco
```

Expected: FAIL，至少一个入口缺少 `semloco`。

- [ ] **Step 3: 扩展所有 choices、usage、正则和 task case**

Python choices 统一改为：

```python
choices=("amp", "distillation", "ppo", "teacher", "ame", "ame_amp", "semloco")
```

`scripts/run_policy_benchmark.sh` 的 case 增加：

```bash
semloco) task_id="Isaac-Go2-Cross-Large-Complex-SemLoco-v0" ;;
```

两个 shell usage/error 文案和 batch validation regex 同步加入 `semloco`。不得改变 batch 的默认 suites、`suite_timeout=1800`、输出目录、`setsid timeout` 或清理函数。

- [ ] **Step 4: 运行静态测试和 shell syntax check**

Run:

```bash
bash -n scripts/run_policy_benchmark.sh scripts/run_policy_benchmark_batch.sh
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation/test_model_adapters.py tests/evaluation/test_policy_benchmark_static.py
```

Expected: shell syntax exit 0；pytest PASS。

- [ ] **Step 5: 提交 SemLoco 接入实现**

```bash
git add evaluation/model_adapters.py scripts/policy_benchmark.py scripts/run_policy_benchmark.sh scripts/run_policy_benchmark_batch.sh tests/evaluation/run_policy_benchmark_1024_smoke.py tests/evaluation/test_model_adapters.py tests/evaluation/test_policy_benchmark_static.py
git commit -m "feat: benchmark SemLoco policy"
```

### Task 3: 回归、真实 1024 smoke 与验证记录

**Files:**
- Create: `notes/log/2026-09-11-policy-benchmark-semloco-integration-smoke.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T305-policy-benchmark.md`

**Interfaces:**
- Consumes: `logs/rsl_rl/cross_large_complex_semloco/2026-09-08_20-08-55/ce29a72/model_19999.pt`。
- Produces: 真实 smoke summary、验证日志、T305 状态和三条正式 batch 命令。

- [ ] **Step 1: 运行全部 evaluation 回归**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q tests/evaluation
```

Expected: 全部 PASS，且没有 collection error。

- [ ] **Step 2: 运行真实 1024 env x 48 transitions smoke**

Run:

```bash
export PATH=/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin:$PATH
export OMNI_KIT_ACCEPT_EULA=Y
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python tests/evaluation/run_policy_benchmark_1024_smoke.py \
  --experiment-type semloco \
  --checkpoint /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/cross_large_complex_semloco/2026-09-08_20-08-55/ce29a72/model_19999.pt \
  --num-envs 1024 \
  --transitions 48 \
  --output-dir /tmp/semloco_policy_benchmark_smoke_20260911 \
  --device cuda:0
```

Expected: exit 0，`valid_windows >= 1`，无 Traceback/OOM/NaN/Inf。

- [ ] **Step 3: 检查输出字段、有限性和残留进程**

Run:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -c 'import json, math, pathlib; p=pathlib.Path("/tmp/semloco_policy_benchmark_smoke_20260911/semloco/complex_mixed/summary.json"); d=json.loads(p.read_text()); assert d["valid_windows"] >= 1; assert all(not isinstance(v, float) or math.isfinite(v) for v in d.values()); print(json.dumps(d, ensure_ascii=False, sort_keys=True))'
pgrep -af 'policy_benchmark.py|run_policy_benchmark_1024_smoke.py|kit/kit|isaacsim' || true
```

Expected: summary 所有顶层浮点值 finite；只允许 `pgrep` 命令自身，无本次 benchmark/Isaac/Kit 残留进程。

- [ ] **Step 4: 写验证日志并更新 T305**

日志必须记录：commit、精确命令、退出码、测试数量、env 数、transition 数、吞吐、episode、valid window、summary 路径、finite 检查及残留进程检查。`notes/todo.md` 和 `notes/todo/T305-policy-benchmark.md` 增加 SemLoco 已接入及真实 smoke 数值；`notes/log/index.md` 增加仓库相对链接。

- [ ] **Step 5: 检查文档并提交验证记录**

Run:

```bash
rg -n "SemLoco|semloco" notes/log/2026-09-11-policy-benchmark-semloco-integration-smoke.md notes/log/index.md notes/todo.md notes/todo/T305-policy-benchmark.md
git diff --check -- notes/log/2026-09-11-policy-benchmark-semloco-integration-smoke.md notes/log/index.md notes/todo.md notes/todo/T305-policy-benchmark.md
```

Expected: 四份文档包含 SemLoco 真实数值，且 diff check exit 0。

```bash
git add notes/log/2026-09-11-policy-benchmark-semloco-integration-smoke.md notes/log/index.md notes/todo.md notes/todo/T305-policy-benchmark.md
git commit -m "docs: record SemLoco benchmark smoke"
```

只暂存本轮写入的文档 hunk；若共享 notes 文件已有用户改动，使用基于补丁的暂存并在提交前核对 `git diff --cached --name-only` 和 `git diff --cached`。
