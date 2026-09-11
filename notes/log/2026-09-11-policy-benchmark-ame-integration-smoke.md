# 2026-09-11 Policy Benchmark AME / AME-AMP 接入与真实 Smoke

## Purpose

把 `ame` 和 `ame_amp` 加入 T305 统一策略 benchmark，验证两类模型复用现有三套 suite、24 帧 valid-only tracking MSE、碰撞/跌倒/位移指标和 batch 进程清理协议。

## Stage

- Workflow: unified policy benchmark
- Model adapters: AME / AME-AMP
- Runtime: Isaac Lab headless evaluation

## Related Todo

- [T305 统一策略评测 Benchmark](../todo/T305-policy-benchmark.md)

## Procedure

1. 用 TDD 扩展 `ModelSpec`、adapter 选择、CLI、single launcher、batch launcher 和 1024-env smoke helper。
2. 对固定 checkpoint 做 iteration、signature、AMP head/discriminator 和 finite tensor 预检。
3. 运行 evaluation 与 AME baseline 回归测试。
4. 分别启动 AME 和 AME-AMP 的 `complex_mixed`、1024 env、48 transitions 真实 headless smoke。
5. 校验 `summary.json`、`progress.json`、`run_manifest.json`、episode shards、MSE 与 episode 指标字段，并检查残留进程。

## Input Conditions

- Branch: `parallelism-amp`
- AME checkpoint: `logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt`
- AME-AMP checkpoint: `logs/rsl_rl/parallelism_tracking_cross_large_complex_ame_amp/2026-09-10_15-42-10/f234425/model_3499.pt`
- Suite: `complex_mixed`
- Environments: `1024`
- Transitions: `48`
- Device argument: `cuda:0`
- Seed: `20260903`
- Smoke output: `eval_output/ame_integration_smoke_20260911/`

## Verification Results

- TDD RED:
  - adapter import failed because `AmeBenchmarkAdapter` / `AmeAmpBenchmarkAdapter` did not exist;
  - shared CLI test failed because `ame` was absent;
  - launcher test failed because `ame_amp` was absent.
- TDD GREEN and regression: `47 passed in 32.52s`.
- Python `py_compile`: exit `0`.
- `bash -n scripts/run_policy_benchmark.sh scripts/run_policy_benchmark_batch.sh`: exit `0`.
- AME checkpoint: `iter=19999`, state dictionary entries `39`, expected AME signature present, all tensor values finite.
- AME-AMP checkpoint: `iter=3499`, model state entries `45`, discriminator state entries `11`, AMP value head present, all checked tensor values finite.
- AME smoke:
  - exit `0`;
  - `episodes=9`;
  - `valid_windows=27`;
  - `elapsed_s=13.817382858134806`;
  - `env_steps_per_second=3557.2583103943157`.
- AME-AMP smoke:
  - exit `0` and therefore passed adapter `full_amp_resume` enforcement;
  - `episodes=8`;
  - `valid_windows=27`;
  - `elapsed_s=11.134257520549`;
  - `env_steps_per_second=4414.483849442747`.
- 两类 summary 的 valid-only MSE 全部 finite。
- 两类 episode shards 均包含 `large_collision_episode`、`small_collision_episode`、`fall`、位移、进度、valid step 和 contact step rate 字段，float 值全部 finite。
- 完成后没有 `policy_benchmark.py`、launcher 或 smoke helper 残留进程。

首次 AME smoke 没有进入 Isaac：外层 helper 使用绝对 Python 路径，但子 shell 从未设置的 `PATH` 中选择到系统 Python，报 `ModuleNotFoundError: isaaclab`。补齐与正式命令一致的 `PATH` 后重跑通过；该失败不涉及 adapter、checkpoint 或仿真逻辑。

## Result

Pass。统一 benchmark 现支持 `amp | distillation | ppo | teacher | ame | ame_amp`。AME 使用专用 map/state wrapper 与 BatchNorm eval inference；AME-AMP 使用运行时 Gym 注册、trajectory manager、专用 runner，并拒绝非完整 AMP resume。

## Conclusion

本次证据证明 AME 与 AME-AMP 的真实 1024-env 接线、checkpoint 加载、24 帧 valid window、共享输出 schema 和进程退出正常。48-step smoke 的短回合数不作为论文行为性能结论。

## Follow-up

分别用六个正式 checkpoint 运行默认三 suite batch，再由 `scripts/summarize_policy_benchmark.py` 生成配对统计和论文表格；不得把 `is_smoke_test=true` 的本次结果混入正式统计。

## Git Refs

- Baseline Ref: `b4121f9`
- Feature Commit: `e816d7a`
- Verified Commit: `e816d7a`
- Key Files:
  - [model_adapters.py](../../evaluation/model_adapters.py)
  - [policy_benchmark.py](../../scripts/policy_benchmark.py)
  - [run_policy_benchmark.sh](../../scripts/run_policy_benchmark.sh)
  - [run_policy_benchmark_batch.sh](../../scripts/run_policy_benchmark_batch.sh)
  - [integration plan](../../docs/superpowers/plans/2026-09-11-ame-policy-benchmark-integration-plan.md)
