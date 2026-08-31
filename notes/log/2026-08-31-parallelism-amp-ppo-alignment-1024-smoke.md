# Parallelism AMP PPO 对齐与 1024 真实 smoke

## 范围

- 分支：`parallelism-amp`
- checkpoint：`logs/rsl_rl/cross_large_complex_ppo/2026-08-26_17-47-24/11d453a/model_19998.pt`
- 实验：`parallelism_tracking_cross_large_complex_amp`
- 规模：1024 env，4 个 PPO iteration，40 rollout steps/env，5 epochs，4 minibatches

## 代码变更

- `ParallelismAMPPPO.update()` 使用 rollout 保存的 `mu/sigma` 计算普通 PPO 同款 adaptive-KL，并更新共享 actor-critic learning rate。
- base critic 与 AMP critic 都使用 clipped value loss；AMP value loss 只统计 active rows。
- actor/base critic/AMP critic 共用一次反向和 `max_grad_norm=1.0`；同时记录裁剪前和裁剪后梯度 norm。
- update 结束恢复普通 PPO 的 `clip_std(min=clip_min_std)`。
- AMP reward 增加 raw/gated 分离统计，runner 通过现有 AMP loss dictionary 写入 TensorBoard。
- 1024 probe 增加 `RUN_REAL_AMP_1024=1` opt-in、日志过滤和 TensorBoard scalar 校验。

## 测试结果

### Focused / regression

```text
28 passed  Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_*.py Go2Pvcnn/tests/tracking/test_parallelism_amp_*.py
44 passed  pure PPO + distillation focused regressions
```

### 真实 1024 smoke

命令使用 `RUN_REAL_AMP_1024=1 NUM_ENVS=1024 MAX_ITERATIONS=4`，probe 返回码 `0`，耗时 `118.80s`，完整迭代 `[0, 1, 2, 3]`，TensorBoard `1` 个 event 文件、`576` 个 scalar。

| 指标 | iteration 0 | iteration 1 | iteration 2 | iteration 3 |
|---|---:|---:|---:|---:|
| `AMP/amp_active_ratio` | 0.4316 | 0.9729 | 0.9737 | 0.9732 |
| `AMP/amp_history_ratio_mean` | 0.7217 | 0.9858 | 0.9870 | 0.9866 |
| `AMP/amp_value_loss` | 116.3183 | 115.8040 | 87.9972 | 21.5718 |
| `AMP/discriminator_loss` | 1.1696 | 1.0502 | 0.6750 | 0.6215 |
| `AMP/approx_kl` | 0.0768 | 0.0060 | 0.0156 | 0.0347 |
| `AMP/actor_critic_grad_norm` | 58.7359 | 77.3255 | 19.8347 | 59.2989 |
| `AMP/actor_critic_grad_norm_clipped` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `Policy/mean_noise_std` | 0.3909 | 0.3898 | 0.3858 | 0.3851 |
| `Perf/collection time (s)` | 13.9809 | 11.8827 | 12.5169 | 12.4781 |
| `Perf/learning_time (s)` | 1.7734 | 0.9282 | 0.9519 | 0.9423 |

日志没有训练异常、OOM、NaN/Inf 或 Traceback；Isaac Sim shutdown 的 USD sourceAsset warning 属于运行时资源告警，不影响进程退出码和训练指标。

## 结论

AMP PPO 对齐和 1024 环境真实 smoke 均通过。当前 4 iteration 中 adaptive-KL 将 learning rate 按普通 PPO 规则调节，actor-critic 梯度均发生统一裁剪；这说明新指标已经能直接暴露“裁剪前梯度过大但参数更新受限”的状态，后续长跑应继续监控这些 TensorBoard 曲线。
