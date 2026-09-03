# 统一策略评测设计文档记录

## Purpose

记录 AMP、Distillation、Pure PPO 和 Parallelism Teacher 的统一仿真评测设计，以及用户确认的场景、命令、指标、并行和统计契约。

## Stage

评测设计 / benchmark harness（尚未进入实现）。

## Related Todo

- [T305 统一策略评测](../todo/T305-policy-benchmark.md)

## Procedure

- 阅读 `notes/index.md`、`notes/todo.md`、T304 branch page、AMP 长跑诊断和现有 play/eval 入口。
- 逐项确认论文主张、三套场景、速度集合、目标点控制、500 layouts、单权重顺序运行和 valid-only MSE。
- 写入 [中文 HTML 设计](../../docs/superpowers/specs/2026-09-03-policy-benchmark-design-zh.html)。
- 执行占位符扫描、`git diff --check`、相对链接存在性检查和 `xmllint --html --noout` 结构检查。

## Input Conditions

- 仿真-only；AMP、Distillation、PPO、Teacher 的用户指定 checkpoint。
- 复杂混合命令为 120 组，专项跑道为大障碍 4/6/8/10 个、小障碍 2.5/5/7.5/10 obstacles/m²。
- Parallelism horizon 为 24 states，replan interval 为 23 transitions。

## Key Results

- HTML 设计文件 361 行、22440 字节；未发现 `TODO`、`TBD`、旧的 87500/99500 回合数或未决占位符。
- 复杂场景回合数修正为 `120×500=60000`；单模型三套 suite 总计约 `72000` episodes。
- `raw/watch-your-step.md` 和 `Go2Pvcnn/scripts/play.py` 相对路径存在。
- `git diff --check` 通过。
- `xmllint` 报告旧 HTML parser 不认识 HTML5 `<main>` 标签；这是工具兼容提示，不是浏览器 HTML5 语法错误。现有仓库 HTML 设计也使用 `<main>`。

## Conclusion

设计已获用户批准，当前没有 runtime 行为或训练结果可以宣称。下一步必须先由用户审阅文档，再创建 implementation plan。

## Git Refs

- Baseline Ref: `8df6931`
- Candidate Ref: working tree design-only
- Key File: `docs/superpowers/specs/2026-09-03-policy-benchmark-design-zh.html`
