# T305 统一策略评测 Benchmark

## Current State

- 中文 HTML 设计已获用户批准并写入 [2026-09-03-policy-benchmark-design-zh.html](../../docs/superpowers/specs/2026-09-03-policy-benchmark-design-zh.html)。
- 目标是用统一、可复现的仿真 benchmark 比较 AMP、Distillation、Pure PPO、AME、AME-AMP、SemLoco，并把 Teacher 标为 planner-conditioned upper bound。
- 已完成评测实现：manifest/protocol、GPU batch 指标、24 帧 valid-only 对齐、模型适配器、断点续跑结果写入器、三套 suite、单权重 CLI、正式汇总和 1024-env smoke。
- 真实 AMP smoke：1024 env x 48 transitions，`env_steps_per_second=5683.82`，`episodes=6`，`valid_windows=27`，无 NaN/Inf/OOM/Traceback。
- 2026-09-06 真实 `small_runway` 排查通过：AMP 1024 env x 48 steps 为 `104 episodes`、`27 valid_windows`、`3222.23 env-steps/s`；Distillation 修复 wrapper 后为 `30 episodes`、`27 valid_windows`、`4131.79 env-steps/s`。此前没有 `small_runway` 记录是因为 CLI 每次只运行一个 suite，运行计划不会自动展开三套 suite。
- Distillation 无结果的根因已定位并修复：PLAY wrapper 漏传 `critic` 与 `distillation_context`，同时 IsaacLab shutdown 的 `SystemExit(0)` 曾掩盖原始 `ValueError`。
- 2026-09-11 已接入 `ame` 与 `ame_amp` 专用 adapter。两者分别用最新最长 checkpoint 完成 1024 env x 48 transitions 真实 smoke：AME 为 `9 episodes`、`27 valid_windows`、`3557.26 env-steps/s`；AME-AMP 为 `8 episodes`、`27 valid_windows`、`4414.48 env-steps/s`。checkpoint、observation/action、MSE 和 episode float 指标均 finite，进程正常退出。
- 2026-09-11 已接入 `semloco`，复用标准 `DefaultBenchmarkAdapter`。最新 `model_19999.pt` 完成 1024 env x 48 transitions 真实 smoke：`6 episodes`、`27 valid_windows`、`5224.88 env-steps/s`；全部 JSON/JSONL 数值 finite，进程正常退出。

## Open Children

- [T305.3](../todo.md#open-leaves)：运行七个正式 checkpoint，生成配对统计和论文表格。

## Closed Children Archive

- 本轮设计确认：120 组复杂 mixed velocity；专项目标点导航、vy=0、23-transition 更新方向；valid-only 只用于 planner tracking MSE。
- T305.4 AME / AME-AMP adapter、六类 launcher 和两次真实 1024-env smoke 已完成。
- T305.5 SemLoco metadata、七类 launcher 和真实 1024-env smoke 已完成。

## Related Logs

- [2026-09-03-policy-benchmark-design.md](../log/2026-09-03-policy-benchmark-design.md)
- [2026-09-06-policy-benchmark-small-runway-distillation-debug.md](../log/2026-09-06-policy-benchmark-small-runway-distillation-debug.md)
- [2026-09-11-policy-benchmark-ame-integration-smoke.md](../log/2026-09-11-policy-benchmark-ame-integration-smoke.md)
- [2026-09-11-policy-benchmark-semloco-integration-smoke.md](../log/2026-09-11-policy-benchmark-semloco-integration-smoke.md)

## Git Refs

- Last Feature Commit: `acc24e7`
- Last Verified Commit: `acc24e7`; regression `34 passed`, real SemLoco 1024-env smoke passed
- Current Work Ref: `parallelism-amp`, formal seven-model paper sweep pending
- Key Files:
  - [统一评测设计](../../docs/superpowers/specs/2026-09-03-policy-benchmark-design-zh.html)
  - [play.py](../../Go2Pvcnn/scripts/play.py)
  - [parallelism_reference_manager.py](../../Go2Pvcnn/tracking/managers/parallelism_reference_manager.py)
  - [model_adapters.py](../../evaluation/model_adapters.py)
  - [batch launcher](../../scripts/run_policy_benchmark_batch.sh)

## Next Step

- 下一步：分别运行七个正式 checkpoint，再用 `scripts/summarize_policy_benchmark.py` 生成配对表格。

## Node Details

### T305.1 统一 benchmark harness

- why-created: 需要让七个模型读取同一 manifest，并在单权重进程内完成批量并行评测。
- contract: 500 个固定 layouts；复杂场景 120 组 command；专项每 density 1500 episodes，三档 vx 各 500。
- dependency: `play.py` 的 runner/wrapper 约定、Parallelism 24-frame/23-transition reference timing。

### T305.2 真实并行验收

- why-created: 评测需要确认 500-env 单次初始化、GPU batch 指标和 valid 24-frame MSE。
- acceptance: 七个 checkpoint 分别通过 smoke，并完成正式三 suite 评测；AME、AME-AMP 与 SemLoco 已完成 1024-env、48-transition 接线验收。

### T305.3 正式论文结果

- why-created: 将六次单权重输出合并为 paired bootstrap、McNemar/Wilcoxon 和最终表格。
- guard: smoke 结果带 `is_smoke_test=true`，不得混入正式汇总。

### T305.4 AME / AME-AMP benchmark adapter

- why-created: AME 与 AME-AMP 的 observation wrapper、runner 和 checkpoint 加载契约不同于原有四类模型，不能复用固定 `OnPolicyRunner` 初始化。
- contract: AME 保持 BatchNorm eval mode；AME-AMP 只接受 `full_amp_resume`；两者复用公共 rollout 和结果 schema。
- evidence: [2026-09-11 integration smoke](../log/2026-09-11-policy-benchmark-ame-integration-smoke.md)。

### T305.5 SemLoco benchmark integration

- why-created: SemLoco 已有独立训练/play 链，但此前不能通过统一 benchmark 与其他策略使用同一 manifest 和指标协议比较。
- contract: 复用标准 `DefaultBenchmarkAdapter`；Shadow Parallelism 只提供 valid-only 24 帧 MSE，不修改 SemLoco observation/action。
- evidence: [2026-09-11 SemLoco integration smoke](../log/2026-09-11-policy-benchmark-semloco-integration-smoke.md)。
