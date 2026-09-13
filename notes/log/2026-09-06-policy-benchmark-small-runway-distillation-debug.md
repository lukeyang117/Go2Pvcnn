# 2026-09-06 policy benchmark small runway / distillation debug

## Purpose

确认为什么 `eval_output` 没有 `small_runway`，并复现用户报告的 Distillation 无结果问题。

## Stage

T305 unified policy benchmark; evaluation wrapper and IsaacLab runtime.

## Related todo

[T305](../todo/T305-policy-benchmark.md)

## Procedure

- 检查 `eval_output/run_index.jsonl`、运行计划和结果目录。
- 使用 `env_isaacsim`、GPU0、真实 IsaacLab headless 运行 AMP `small_runway`：16 env/1 step，再 1024 env/48 steps。
- 使用真实 Distillation checkpoint 运行 `small_runway`：32 env/1 step，修复前后对比；修复后再运行 1024 env/48 steps。
- 用临时 `SimulationContext` STOP 回调探针保留初始化异常调用栈。

## Evidence

### Missing suite

`eval_output/run_index.jsonl` 在修复前没有任何 `small_runway` 记录或目录。`run_policy_benchmark.sh` 每次只转发一个必选 `--suite`，`run_plan_2026-09-04.json` 只是协议清单，不会自动执行三套 suite；此前只启动了 `complex_mixed` 和 `large_runway`。

### AMP small runway

- 16 env/1 step: exit `0`, summary written.
- 1024 env/48 steps: exit `0`, `episodes=104`, `valid_windows=27`, `env_steps_per_second=3222.2337`; no CUDA illegal access, NaN/Inf, OOM, or traceback.

### Distillation root cause

Before the wrapper fix, the real traceback was:

```text
ValueError: Hybrid distillation training requires extras['observations']['critic']
```

`Go2Pvcnn/scripts/play.py::_format_observations` returned only `teacher` for `student_state`, omitting `critic` and `distillation_context`. `policy_benchmark.py` then constructed `HybridDistillationPPO`, which requires all three observation groups. IsaacLab shutdown raised `SystemExit(0)` in the outer `finally`, masking the original exception and leaving no `summary.json`.

### Distillation after fix

- 32 env/1 step: exit `0`; student/critic/teacher networks initialized with input dimensions `1069/1072/1103`.
- 1024 env/48 steps: exit `0`, `episodes=30`, `valid_windows=27`, `env_steps_per_second=4131.7946`; summary written to `/tmp/distillation_small_runway_after_wrapper_fix_1024/distillation/small_runway/summary.json`.

## Changes

- `Go2Pvcnn/scripts/play.py`: export flattened `critic` and `distillation_context` alongside `teacher` for Distillation observations.
- `scripts/policy_benchmark.py`: preserve an active exception when IsaacLab app shutdown raises `SystemExit(0)`.
- `tests/evaluation/test_policy_benchmark_static.py`: regress both contracts.

## Verification

- RED test for app-close masking: failed with missing helper.
- GREEN evaluation/distillation regression tests: `50 passed` (including the new `4` evaluation tests).
- Real IsaacLab AMP and Distillation runs above completed without numerical/runtime errors.

## Conclusion

`small_runway` was absent from prior results because it was never launched; the benchmark CLI is single-suite per process. Distillation had a separate wrapper contract bug that prevented any suite from writing results and was hidden by shutdown handling. The small-runway scene itself is valid at 1024 envs for the tested 80-obstacle first condition.

## Git refs

- Baseline ref: `parallelism-amp`
- Candidate ref: working tree after wrapper/error-propagation fix
- Key files: `Go2Pvcnn/scripts/play.py`, `scripts/policy_benchmark.py`, `tests/evaluation/test_policy_benchmark_static.py`
