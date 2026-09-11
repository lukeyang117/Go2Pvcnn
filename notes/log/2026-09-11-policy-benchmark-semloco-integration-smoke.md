# 2026-09-11 Policy Benchmark SemLoco 接入与真实 Smoke

## Purpose

把 `semloco` 加入 T305 统一策略 benchmark，验证它复用标准 RSL-RL adapter、现有三套 suite、碰撞/跌倒/位移指标和 valid-only 24 帧 tracking MSE。

## Related Todo

- [T305 统一策略评测 Benchmark](../todo/T305-policy-benchmark.md)

## Input Conditions

- Branch: `parallelism-amp`
- Feature commit: `acc24e7`
- Checkpoint: `logs/rsl_rl/cross_large_complex_semloco/2026-09-08_20-08-55/ce29a72/model_19999.pt`
- Checkpoint SHA256: `89117544e2836633f435e33c5990fadcce5cf54daa6c272ad647b6f8ef26d8ea`
- Suite: `complex_mixed`
- Environments: `1024`
- Transitions: `48`
- Device: `cuda:0`
- Seed: `20260903`
- Output: `/tmp/semloco_policy_benchmark_smoke_20260911/semloco/complex_mixed/`

## Procedure

1. 用 TDD 添加 SemLoco metadata、PLAY cfg 映射和四类命令入口。
2. 运行完整 `tests/evaluation` 回归。
3. 使用 smoke helper 启动真实 1024 环境、48 transitions 的 headless Isaac Lab。
4. 递归解析 `summary.json`、`run_manifest.json`、`progress.json` 和全部 episode JSONL，检查所有浮点值 finite。
5. 检查 benchmark、Isaac/Kit 残留进程和 GPU 显存释放。

真实运行命令：

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

## Verification Results

- Baseline regression before implementation: `32 passed in 6.99s`.
- Task 1 RED: `ValueError: unknown experiment type: semloco`.
- Task 1 GREEN: `5 passed in 0.19s`.
- Task 2 RED: launcher source did not contain `semloco`.
- Focused GREEN: `15 passed in 5.71s`; both shell syntax checks exited `0`.
- Full evaluation regression after implementation: `34 passed in 4.68s`.
- Real smoke: exit `0`.
- `elapsed_s=9.407292505726218`.
- `env_steps_per_second=5224.8827141370575`.
- `episodes=6` and six episode shards.
- `valid_windows=27`.
- Mean valid window ratio: `0.9448784722222222`.
- Mean joint position/velocity MSE: `0.041307784616947174` / `6.907941553327772`.
- Mean root position/rotation MSE: `0.0004350300123742609` / `0.0005792907797041591`.
- Mean root linear/angular velocity MSE: `0.045301247112177034` / `0.11868954781028959`.
- 所有 JSON/JSONL 浮点值 finite。
- Episode shards 包含 `large_collision_episode`、`small_collision_episode`、`fall`、`traversed_path_m`、`command_progress_m`、`valid_step_count` 和 `contact_step_rate`。
- 48-step smoke 的六个早停回合中：大障碍碰撞率 `1.0`、小障碍碰撞率 `0.0`、跌倒率 `1.0`、平均位移 `0.07422229647636414m`。这些短 smoke 数值只验证字段和数据流，不作为论文性能结果。
- 完成后 GPU 为 `1 MiB`、利用率 `0%`，没有本次 `policy_benchmark.py`、smoke helper 或 Isaac/Kit 残留进程。

## Result

Pass。统一 benchmark 现在支持 `amp | distillation | ppo | teacher | ame | ame_amp | semloco`。SemLoco 通过 `DefaultBenchmarkAdapter` 加载，Shadow Parallelism 只生成 valid-only MSE 参考，不改变 SemLoco 策略观测和动作。

## Follow-up

正式结果必须使用不带 `--smoke-test` 的 batch launcher 完成三套 suite；本次 `is_smoke_test=true` 的输出不得混入论文统计。

## Git Refs

- Design commit: `2c562b1`
- Plan commit: `ef85fa4`
- Feature commit: `acc24e7`
- Key files:
  - [model_adapters.py](../../evaluation/model_adapters.py)
  - [policy_benchmark.py](../../scripts/policy_benchmark.py)
  - [batch launcher](../../scripts/run_policy_benchmark_batch.sh)
  - [design](../../docs/superpowers/specs/2026-09-11-semloco-policy-benchmark-integration-design.md)
  - [implementation plan](../../docs/superpowers/plans/2026-09-11-semloco-policy-benchmark-integration-plan.md)
