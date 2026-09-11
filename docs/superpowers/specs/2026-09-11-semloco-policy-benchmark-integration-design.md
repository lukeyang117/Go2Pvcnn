# SemLoco 统一策略评测接入设计

日期：2026-09-11

状态：已获用户批准

## 目标

在现有统一 policy benchmark 中新增 `semloco` 实验类型。SemLoco 必须与 AMP、Distillation、PPO、Teacher、AME 和 AME-AMP 使用相同的三套 suite、场景 manifest、碰撞/跌倒/位移指标和 valid-only 24 帧 tracking MSE。

本次不修改 SemLoco 的训练行为、网络结构、reward、planner、checkpoint 格式或现有 play 入口。

## 固定评测权重

使用当前训练 iteration 最大的 SemLoco checkpoint：

```text
logs/rsl_rl/cross_large_complex_semloco/2026-09-08_20-08-55/ce29a72/model_19999.pt
```

## 架构决策

SemLoco 复用 `DefaultBenchmarkAdapter`，不新增专用 adapter。

其运行契约与现有 SemLoco play 入口保持一致：

- Experiment：`cross_large_complex_semloco`
- Gym task：`Isaac-Go2-Cross-Large-Complex-SemLoco-v0`
- PLAY cfg：`SemlocoCrossLargeComplexEnvCfg_PLAY`
- Observation kind：`deployment`
- Wrapper：现有 `Go2Pvcnn.scripts.play._make_env_wrapper`
- Runner：标准 `rsl_rl.runners.OnPolicyRunner`
- 训练配置：`get_train_cfg("cross_large_complex_semloco")`
- 加载：`runner.load(checkpoint, load_optimizer=False)`

SemLoco 当前没有专用 wrapper 或 runner；新增空壳专用 adapter 会重复默认逻辑，因此不采用。

## 数据流

```text
--experiment-type semloco
  -> resolve_model_spec()
  -> DefaultBenchmarkAdapter
  -> SemlocoCrossLargeComplexEnvCfg_PLAY
  -> 标准 wrapper / OnPolicyRunner 加载 checkpoint
  -> policy(obs) 输出 12 维动作
  -> 环境并行 step
  -> 公共 episode metrics
  -> 旁路 Shadow Parallelism reference
  -> valid-only 24 帧 joint / foot / root MSE
  -> ResultWriter 写入公共 schema
```

Shadow Parallelism manager 只负责生成评测参考轨迹和 MSE。其轨迹不得写入 SemLoco observation，也不得改变策略动作、环境动力学或 episode 终止逻辑。无 valid 轨迹时不记录 tracking window；恢复规划后必须重新积满连续 24 帧才恢复 MSE 统计。

## CLI 与输出

以下入口新增 `semloco` 合法类型：

- `scripts/policy_benchmark.py`
- `scripts/run_policy_benchmark.sh`
- `scripts/run_policy_benchmark_batch.sh`
- `tests/evaluation/run_policy_benchmark_1024_smoke.py`

正式 batch 默认依次运行：

```text
complex_mixed -> large_runway -> small_runway
```

三套 suite 位于同一次自动命名目录：

```text
eval_output/<timestamp>_semloco_model_19999_<checkpoint-sha256>/
  run_manifest.json
  complex_mixed/
  large_runway/
  small_runway/
```

目录和 manifest 记录执行时间、模型类型、RL task、checkpoint 绝对路径及 SHA256。每个 suite 继续生成 episode shard、progress 和 summary；现有汇总工具通过公共 schema 读取 SemLoco，不增加模型专用指标格式。

## 错误处理与进程生命周期

- checkpoint 必须是明确存在的文件，否则在启动 Isaac Lab 前失败。
- 未知 experiment type、缺失 task/cfg 映射、checkpoint 与网络维度不匹配均立即报错。
- 公共 finite check 继续拒绝非有限 observation、action、reference、MSE 或 episode 浮点指标。
- 任一 suite 非零退出或未生成 `summary.json` 时，batch 终止且不启动后续 suite。
- 每个 suite 结束后继续执行现有进程组清理和 GPU 释放等待，避免前一个 Isaac/Kit 进程阻塞下一场景。

## 文件边界

| 文件 | 改动 |
|---|---|
| `evaluation/model_adapters.py` | 增加 SemLoco metadata 与 PLAY cfg 映射，复用默认 adapter |
| `scripts/policy_benchmark.py` | CLI choices 接受 `semloco` |
| `scripts/run_policy_benchmark.sh` | 增加 SemLoco task/index 映射 |
| `scripts/run_policy_benchmark_batch.sh` | 增加 SemLoco 合法类型 |
| `tests/evaluation/run_policy_benchmark_1024_smoke.py` | smoke helper 接受 SemLoco |
| `tests/evaluation/*` | 增加 metadata、adapter、CLI 与 shell 回归 |

## 测试与验收

### 不启动 Isaac Lab 的测试

- CLI、单次 launcher、batch launcher 和 smoke helper 接受 `semloco`，仍拒绝未知类型。
- `resolve_model_spec("semloco", checkpoint)` 返回正确 experiment、task、observation 和 adapter 类型。
- `build_benchmark_adapter()` 为 SemLoco 返回 `DefaultBenchmarkAdapter`。
- PLAY mapping 返回 `SemlocoCrossLargeComplexEnvCfg_PLAY`。
- 原有六类模型和全部 evaluation 回归继续通过。

### 真实 Isaac Lab 测试

使用固定 checkpoint 执行 `complex_mixed` 的 `1024 env x 48 transitions` smoke：

- 进程退出码为 0。
- 至少生成一个完整 valid 24 帧 tracking window。
- observation、action、reference、MSE 和 episode 浮点指标全部 finite。
- 生成 episode shard、progress、run manifest 和 summary。
- summary 包含吞吐、episode、valid window、碰撞、跌倒、位移和 valid-only MSE 字段。
- 进程正常退出，且不遗留 benchmark/Isaac/Kit 子进程。

完成 smoke 后提供 SemLoco、AME 和 AME-AMP 三条不带 `--smoke-test` 的正式 batch 命令。三条命令都使用 `1024` 环境、`cuda:0`、headless，并由脚本自动生成输出目录。

## 完成标准

SemLoco 能以与现有模型一致的命令形态运行统一 benchmark；最新 `model_19999.pt` 通过真实 1024 环境 smoke；输出可由现有汇总工具直接读取；AME 与 AME-AMP 行为保持不变。
