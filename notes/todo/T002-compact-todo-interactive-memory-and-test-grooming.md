# T002 compact-todo interactive memory and test grooming

## Current State

- The user approved the redesign direction for `compact-todo` in conversation.
- The redesign turns `compact-todo` into a direction-driven grooming session rather than a static compaction checklist.
- The scope includes the whole repository memory system:
  - `notes/todo`
  - `notes/log`
  - archive entrypoints
- The redesign also adds full-tree test review for `Go2Pvcnn/tests/`.
- Destructive actions remain user-controlled:
  - archive
  - delete
  - merge
  - test deletion
- The written spec is being created and the next gate is user review of the spec file.

## Open Children

- [T002a](#t002a-written-spec-review-gate): review the written `compact-todo` redesign spec before implementation planning.

## Closed Children Archive

- none

## Related Logs

- [2026-05-09-1812-compact-todo-interactive-design.md](../log/2026-05-09-1812-compact-todo-interactive-design.md)

## Git Refs

- Last Feature Commit: `pending (design stage only)`
- Last Verified Commit: `pending (design stage only)`
- Current Work Ref: `working tree on top of 130c635 (2026-05-09 18:12 +0800); compact-todo redesign spec with unrelated planner/notes dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md)
  - [../../.agents/skills/compact-todo/SKILL.md](../../.agents/skills/compact-todo/SKILL.md)
  - [../todo.md](../todo.md)
  - [../log/index.md](../log/index.md)
  - [2026-05-09-1812-compact-todo-interactive-design.md](../log/2026-05-09-1812-compact-todo-interactive-design.md)

## Next Step

- Ask the user to review [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md).
- If the user approves the written spec, invoke the `writing-plans` workflow for implementation planning.

## Node Details

### T002a written spec review gate

- status: doing
- why-created:
  - the conversation already approved the design direction
  - the written spec still needs explicit user review before implementation planning
- evidence:
  - [2026-05-09-1812-compact-todo-interactive-design.md](../log/2026-05-09-1812-compact-todo-interactive-design.md)
- acceptance:
  - the written spec captures direction-driven memory grooming
  - the written spec captures full `Go2Pvcnn/tests/` review
  - destructive actions remain user-approved
  - auto-splitting stays limited to non-destructive restructuring
