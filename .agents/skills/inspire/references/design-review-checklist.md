# Design Review Checklist

Every inspire design draft requires a focused subagent review. Keep the review narrow: requirement coverage, acceptance-indicator coverage, and ambiguity that could mislead implementation.

## Mandatory Review Questions

- Did the design capture the requirement points already discussed with the user?
- Did the design capture the intended workflow gates and forbidden transitions?
- Did the design preserve manual `/inspire` triggering and discussion-first behavior?
- Did the design keep discussion choices limited to `继续分析 / 生成设计 / 结束`?
- Did the design forbid a direct jump from discussion to todo?
- Did the design explicitly enter `brainstorming` when the user chooses `生成设计`?
- Did the design require subagent review before user-facing design approval?
- Did the design path require syncing design-state memory in both `notes/todo.md` and the relevant branch page under `notes/todo/` before the design is presented as ready?
- Did the design use todo-first breakdown instead of a standalone implementation plan?
- Did the design keep the primary agent on orchestration/notes/log/review only?
- Did the design keep subagents on code changes and testing during execution?
- Did the design include testing and acceptance indicators that would catch the wrong implementation?
- Is any requirement or acceptance indicator still ambiguous enough to cause the wrong implementation?

## Required Review Output

The subagent must answer with exactly these sections:

- `已落实的需求点`
- `缺失或不清楚的需求点`
- `已落实的测试/验收指标`
- `缺失或不清楚的测试/验收指标`
- `建议修改项`
- `是否建议通过本轮 design`

## Re-Review Rule

If the primary agent makes a substantive change after review, run this review again before presenting the design as ready for user review.
