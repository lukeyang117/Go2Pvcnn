# T305 统一策略评测 Benchmark

## Current State

- 中文 HTML 设计已获用户批准并写入 [2026-09-03-policy-benchmark-design-zh.html](../../docs/superpowers/specs/2026-09-03-policy-benchmark-design-zh.html)。
- 目标是用统一、可复现的仿真 benchmark 比较 AMP、Distillation、Pure PPO，并把 Teacher 标为 planner-conditioned upper bound。
- 已完成评测实现：manifest/protocol、GPU batch 指标、24 帧 valid-only 对齐、模型适配器、断点续跑结果写入器、三套 suite、单权重 CLI、正式汇总和 1024-env smoke。
- 真实 AMP smoke：1024 env x 48 transitions，`env_steps_per_second=5683.82`，`episodes=6`，`valid_windows=27`，无 NaN/Inf/OOM/Traceback。

## Open Children

- [T305.3](../todo.md#open-leaves)：运行四个正式 checkpoint，生成配对统计和论文表格。

## Closed Children Archive

- 本轮设计确认：120 组复杂 mixed velocity；专项目标点导航、vy=0、23-transition 更新方向；valid-only 只用于 planner tracking MSE。

## Related Logs

- [2026-09-03-policy-benchmark-design.md](../log/2026-09-03-policy-benchmark-design.md)

## Git Refs

- Last Feature Commit: pending
- Last Verified Commit: working tree verified; commit after user review
- Current Work Ref: `parallelism-amp` working tree, benchmark implementation
- Key Files:
  - [统一评测设计](../../docs/superpowers/specs/2026-09-03-policy-benchmark-design-zh.html)
  - [play.py](../../Go2Pvcnn/scripts/play.py)
  - [parallelism_reference_manager.py](../../Go2Pvcnn/tracking/managers/parallelism_reference_manager.py)

## Next Step

- 下一步：分别运行四个正式 checkpoint，再用 `scripts/summarize_policy_benchmark.py` 生成配对表格。

## Node Details

### T305.1 统一 benchmark harness

- why-created: 需要让四个模型读取同一 manifest，并在单权重进程内完成批量并行评测。
- contract: 500 个固定 layouts；复杂场景 120 组 command；专项每 density 1500 episodes，三档 vx 各 500。
- dependency: `play.py` 的 runner/wrapper 约定、Parallelism 24-frame/23-transition reference timing。

### T305.2 真实并行验收

- why-created: 评测需要确认 500-env 单次初始化、GPU batch 指标和 valid 24-frame MSE。
- acceptance: 四个 checkpoint 分别通过 16-env smoke，并分别通过 500-env、48-transition 的真实验收。

### T305.3 正式论文结果

- why-created: 将四次单权重输出合并为 paired bootstrap、McNemar/Wilcoxon 和最终表格。
- guard: smoke 结果带 `is_smoke_test=true`，不得混入正式汇总。
