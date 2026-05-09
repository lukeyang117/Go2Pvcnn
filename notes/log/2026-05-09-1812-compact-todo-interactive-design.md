# compact-todo interactive memory and test grooming design

## Purpose

- Record the approved redesign direction for project-local `compact-todo`.
- Convert the skill from a static compaction checklist into a direction-driven memory and test grooming workflow.

## Stage

- repository tooling / local skill design

## Related Todo

- [T002/T002a](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002a-written-spec-review-gate)

## Command / Procedure

- Read repository constraints and memory entrypoints:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
  - [../todo.md](../todo.md)
  - [../todo/README.md](../todo/README.md)
  - [index.md](index.md)
- Read the current `compact-todo` skill definition:
  - [../../.agents/skills/compact-todo/SKILL.md](../../.agents/skills/compact-todo/SKILL.md)
- Reviewed prior notes-workflow compact logs:
  - [2026-04-30-1456-notes-compact-todo.md](2026-04-30-1456-notes-compact-todo.md)
  - [2026-04-30-1450-t200-branch-compact.md](2026-04-30-1450-t200-branch-compact.md)
- Brainstormed the redesign in conversation and confirmed:
  - future direction is provided in plain language
  - whole notes memory tree is in scope
  - open nodes may auto-split into deeper `.md` pages
  - archive/delete/merge actions require user approval
  - uncertain cases default to asking the user
  - the full `Go2Pvcnn/tests/` tree is scanned every compact session
  - stale tests are reviewed against current code, logs, and future direction
  - test candidates are grouped by module first, then candidate type
- Wrote the design spec at [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md).
- Updated repository memory:
  - [../todo.md](../todo.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [index.md](index.md)

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Baseline ref: `130c635`
- Current work ref: `working tree on top of 130c635; unrelated planner/notes dirt present`
- Design output:
  - [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md)

## Key Metrics

- Design sections written: `16`
- Memory surfaces updated: `3`
  - [../todo.md](../todo.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [index.md](index.md)
- Required workflow additions captured: `2`
  - interactive whole-tree memory grooming
  - full `Go2Pvcnn/tests/` asset review

## Result

- Pass

## Conclusion

- The `compact-todo` redesign is now written as a formal spec and synced into repository memory.
- The next gate is user review of the written spec before implementation planning begins.

## Follow-up

- Ask the user to review [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md).
- If the user approves the written spec, invoke the `writing-plans` workflow for implementation planning.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: `working tree on top of 130c635 (2026-05-09 18:12 +0800); compact-todo design spec and notes update`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [index.md](index.md)
