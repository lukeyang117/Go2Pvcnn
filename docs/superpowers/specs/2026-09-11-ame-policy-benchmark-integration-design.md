# AME / AME-AMP 统一策略评测接入设计

日期：2026-09-11

状态：已获用户批准

## 目标

在现有统一 policy benchmark 中新增 `ame` 和 `ame_amp` 两种实验类型。两者必须与已有 AMP、Distillation、PPO、Teacher 使用相同的三套 suite、condition manifest、碰撞/跌倒/位移指标和 valid-only 24 帧 tracking MSE，且继续由 batch launcher 顺序启动三个独立 Isaac Lab 进程。

本次不修改 AME 或 AME-AMP 的训练行为、网络结构、reward、planner、checkpoint 格式和现有 play 入口。

## 固定评测权重

使用当前训练 iteration 最大且已验证 checkpoint 结构完整的权重：

| 类型 | Checkpoint | 验证 |
|---|---|---|
| `ame` | `logs/rsl_rl/cross_large_complex_ame/2026-09-08_15-40-23/bfb8b7f/model_19999.pt` | `iter=19999`，包含 AME architecture signature |
| `ame_amp` | `logs/rsl_rl/parallelism_tracking_cross_large_complex_ame_amp/2026-09-10_15-42-10/f234425/model_3499.pt` | `iter=3499`，同时包含 AMP value head、discriminator 和两个 optimizer state |

## 架构决策

新增两个专用 adapter 类，把模型特有的环境、wrapper、runner 和 checkpoint 加载封装在统一接口后面：

### `AmeBenchmarkAdapter`

- 环境配置：`AmeCrossLargeComplexEnvCfg_PLAY`
- Gym task：复用 `Isaac-Go2-Cross-Large-Complex-PPO-v0`，创建时显式传入 AME cfg
- Wrapper：`AmeRslRlEnvWrapper`
- Runner：`AmeOnPolicyRunner`
- 训练配置：`get_ame_train_cfg()`
- 加载：`runner.load(checkpoint, load_optimizer=False, keep_std=True)`
- 推理前调用 eval/inference policy，确保 AME BatchNorm 使用保存的 running statistics

### `AmeAmpBenchmarkAdapter`

- 环境配置：`AmeParallelismAmpCrossLargeComplexEnvCfg`
- Gym task：运行时幂等注册 `AME_AMP_ENV_ID`
- Entry point：`tracking.amp_env:ParallelismAmpEnv`
- Wrapper：`AmeRslRlEnvWrapper`
- Runner：`AmeAmpOnPolicyRunner`
- 训练配置：`get_ame_amp_train_cfg()`
- 加载：`runner.load_amp_checkpoint(checkpoint, keep_std=True)`
- 只接受 `full_amp_resume`，拒绝缺少 AMP head 或 discriminator 的 legacy/incomplete checkpoint
- 环境创建后按现有 AME-AMP play 契约 attach trajectory manager

两个类向 benchmark rollout 暴露相同结果：`base_env`、wrapped env、deterministic inference policy 和初始 policy observation。共享 rollout 不感知 AME 网络内部结构。

## 数据流

```text
--experiment-type ame | ame_amp
  -> resolve_model_spec()
  -> 创建对应 BenchmarkAdapter
  -> adapter 创建 env / wrapper / runner
  -> adapter 严格加载 checkpoint
  -> 公共 rollout：policy(obs) -> env.step(action)
  -> 公共 shadow Parallelism reference
  -> 公共 episode metrics + valid 24-frame MSE
  -> ResultWriter 写入相同 schema
```

AME 的 actor observation 仍为六通道地图加 45 维 state；AME-AMP 的 actor observation与 AME 相同。Shadow planner 只用于评测 reference 和 MSE，不得写入 AME actor observation。AME-AMP 环境自身 planner 生命周期保持原状，但评测指标读取统一的 reference manager。

## CLI 与输出

以下入口同时扩展为六种类型：

```text
amp | distillation | ppo | teacher | ame | ame_amp
```

- `scripts/policy_benchmark.py`
- `scripts/run_policy_benchmark.sh`
- `scripts/run_policy_benchmark_batch.sh`
- `evaluation/model_adapters.py`

默认 batch 仍依次运行：

```text
complex_mixed -> large_runway -> small_runway
```

输出目录沿用：

```text
eval_output/<timestamp>_<experiment-type>_<weight>_<sha256>/
```

每种模型分别记录 RL task、绝对 checkpoint 路径、SHA256、开始/结束时间和退出码。现有汇总脚本接受任意模型标签，因此只要 episode/summary schema 不变，不需要新增 AME 专用指标格式。

## 错误处理

- AME checkpoint signature 或 observation/action 维度不匹配时立即失败。
- AME-AMP 缺少 AMP head、discriminator 或包含非有限 tensor 时立即失败。
- BatchNorm 必须处于 eval 模式；不得在评测 rollout 更新 running statistics。
- 任一 suite 非零退出或没有生成 `summary.json` 时停止后续 suite。
- suite 结束后继续清理进程组并等待 GPU 释放，避免下一场景卡死。
- 公共 finite check 继续拒绝非有限 observation/action。

## 文件边界

| 文件 | 改动 |
|---|---|
| `evaluation/model_adapters.py` | 扩展 metadata；新增 `AmeBenchmarkAdapter` 与 `AmeAmpBenchmarkAdapter` 及统一构建入口 |
| `scripts/policy_benchmark.py` | 扩展 CLI，并通过 adapter 创建模型运行时；保留共享 rollout/metrics |
| `scripts/run_policy_benchmark.sh` | 增加 `ame`、`ame_amp` task/index 映射 |
| `scripts/run_policy_benchmark_batch.sh` | 增加两种合法 experiment type |
| `tests/evaluation/*` | 增加 CLI、adapter、runner/load、shell 和回归测试 |

## 测试与验收

### 不启动 Isaac Lab 的测试

- CLI 接受 `ame`、`ame_amp`，仍拒绝未知类型。
- 两个 ModelSpec 返回正确 experiment、task 和 adapter 类型。
- AME adapter 选择 AME cfg/wrapper/runner/config/load。
- AME-AMP adapter幂等注册 task，选择 AMP env/runner/config/load，并拒绝非完整 checkpoint。
- 两个 shell launcher 接受新类型，生成正确自动输出目录和 run index。
- 原有四类模型 adapter 和全部 evaluation 测试继续通过。

### 真实 Isaac Lab 测试

使用上述两个固定 checkpoint，分别执行 `1024 env` smoke：

- 至少运行 48 transitions，产生第一个完整24帧 valid tracking window。
- observation、action、reference、MSE 和 checkpoint tensor 全部 finite。
- 生成 episode shard、`progress.json`、`run_manifest.json`、`summary.json`。
- 碰撞、跌倒、位移和 valid-only MSE 字段存在且数值有效。
- 进程正常退出，不遗留 AME/AME-AMP/Isaac/Kit 子进程。

完成 smoke 后提供两个正式全量 batch 命令；命令不包含 `--smoke-test`，默认运行三个 suite。

## 完成标准

`ame` 和 `ame_amp` 能使用与现有四类模型完全一致的命令形态运行统一 benchmark，两个最新最长训练权重均通过真实 Isaac Lab smoke，并能生成可由现有汇总工具读取的同 schema 结果。
