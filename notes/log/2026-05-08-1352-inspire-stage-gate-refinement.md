# inspire stage-gate refinement

## Purpose

- Refine the project-local `/inspire` skill so design and todo remain explicit stop points, but explicit advancement language starts the next stage immediately and execution no longer requires any further user approval after stage entry.

## Stage

- repository tooling / local skill behavior refinement

## Related Todo

- [T001](../todo/T001-inspire-skill-design.md)

## Command / Procedure

- Re-read repository constraints and active memory:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
  - [../todo.md](../todo.md)
  - [../todo/T001-inspire-skill-design.md](../todo/T001-inspire-skill-design.md)
  - [index.md](index.md)
- Re-read the live `/inspire` skill package:
  - [../../.agents/skills/inspire/SKILL.md](../../.agents/skills/inspire/SKILL.md)
  - [../../.agents/skills/inspire/references/intake-modes.md](../../.agents/skills/inspire/references/intake-modes.md)
  - [../../.agents/skills/inspire/references/analysis-lenses.md](../../.agents/skills/inspire/references/analysis-lenses.md)
  - [../../.agents/skills/inspire/references/design-template.md](../../.agents/skills/inspire/references/design-template.md)
  - [../../.agents/skills/inspire/references/design-review-checklist.md](../../.agents/skills/inspire/references/design-review-checklist.md)
  - [../../.agents/skills/inspire/references/todo-write-contract.md](../../.agents/skills/inspire/references/todo-write-contract.md)
  - [../../.agents/skills/inspire/references/delegation-contract.md](../../.agents/skills/inspire/references/delegation-contract.md)
- Folded in the clarified workflow requirements from live user feedback:
  - discussion replies should say what the next part will cover
  - design and todo stay as real stop points
  - explicit `继续` / `下一阶段` after design or todo should start the next stage immediately
  - execution approval is stage-level, after which no further user approval is required
  - the primary agent should stay alive and autonomous between normal subagent cycles
- Patched the skill package, the original inspire design spec, and the T001 repository memory/log surfaces to match the new stage-gate contract.
- Ran fresh grep and readback verification focused on the removed post-entry approval wording and the new stage-advance wording.

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Baseline ref: `130c635`
- Current work ref: `working tree on top of 130c635 (2026-05-08 13:52 +0800); inspire stage-gate refinement with unrelated planner/viewer/plugin dirt present`
- Existing unrelated workspace dirt remained outside the owned file set.

## Key Metrics

- Patched files: `11`
- Explicit stage boundaries preserved: `2` (`design`, `todo`)
- Removed post-entry execution approval loop: `1`
- Fresh verification checks run by primary agent: `4`
  - grep for removed `specific todo leaf` / post-entry approval wording
  - grep for new `继续` / `下一阶段` stage-advance wording
  - grep for continued `stop and wait` boundaries
  - targeted readback of patched skill and reference files

## Result

- Pass

## Conclusion

- `/inspire` now keeps the design and todo stage stops, but it no longer re-asks for the same approval after the user explicitly advances to the next stage.
- Execution now uses stage-level approval with full autonomy after entry: once implementation starts, the primary agent chooses ready leaves, waits for subagent results, updates todo/log state, and keeps orchestrating until complete or objectively blocked.
- The discussion contract now also requires telling the user what the next part of the conversation will cover.

## Follow-up

- Run one live `/inspire` session to confirm the conversational pacing feels right in practice, especially at the `design -> todo -> execution` boundaries.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: `working tree on top of 130c635 (2026-05-08 13:52 +0800); inspire stage-gate refinement and notes update`
- Key Files:
  - [../../.agents/skills/inspire/SKILL.md](../../.agents/skills/inspire/SKILL.md)
  - [../../.agents/skills/inspire/references/intake-modes.md](../../.agents/skills/inspire/references/intake-modes.md)
  - [../../.agents/skills/inspire/references/analysis-lenses.md](../../.agents/skills/inspire/references/analysis-lenses.md)
  - [../../.agents/skills/inspire/references/design-template.md](../../.agents/skills/inspire/references/design-template.md)
  - [../../.agents/skills/inspire/references/design-review-checklist.md](../../.agents/skills/inspire/references/design-review-checklist.md)
  - [../../.agents/skills/inspire/references/todo-write-contract.md](../../.agents/skills/inspire/references/todo-write-contract.md)
  - [../../.agents/skills/inspire/references/delegation-contract.md](../../.agents/skills/inspire/references/delegation-contract.md)
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md)
  - [../todo/T001-inspire-skill-design.md](../todo/T001-inspire-skill-design.md)
  - [index.md](index.md)
