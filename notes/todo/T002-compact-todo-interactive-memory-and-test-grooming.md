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
- The written spec is approved by the user for implementation and the first implementation pass is now landed in `.agents/skills/compact-todo/SKILL.md`.
- A focused implementation plan is written for traceability even though the user asked to proceed directly with the change.
- One live branch-compact pressure pass is complete; grouped stale-test review is the remaining live-usage gap.
- A follow-up hardening pass now makes tree-preserving child compaction and Obsidian index paths explicit repository-memory constraints.

## Open Children

- [T002b](#t002b-live-usage-pressure-verification): pressure-test the implemented skill in a real compact session.

## Closed Children Archive

- T002c done: child-node compaction now preserves tree-shaped parent/child navigation and Obsidian index paths.

## Related Logs

- [2026-05-09-1812-compact-todo-interactive-design.md](../log/2026-05-09-1812-compact-todo-interactive-design.md)
- [2026-05-09-1834-compact-todo-implementation.md](../log/2026-05-09-1834-compact-todo-implementation.md)
- [2026-05-09-1848-t100-branch-compact-except-t116.md](../log/2026-05-09-1848-t100-branch-compact-except-t116.md)
- [2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md](../log/2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md)
- [2026-05-10-2043-compact-todo-together-planner-test-cleanup-scan.md](../log/2026-05-10-2043-compact-todo-together-planner-test-cleanup-scan.md)

## Git Refs

- Last Feature Commit: `pending (implementation staged, not yet committed)`
- Last Verified Commit: `readback + grep verification at 2026-05-10 21:39 +0800`
- Current Work Ref: `working tree on top of 979b2b5 (2026-05-10 21:39 +0800); compact-todo tree/index hardening with unrelated planner/notes dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md)
  - [../../docs/superpowers/plans/2026-05-09-compact-todo-interactive-memory-and-test-grooming.md](../../docs/superpowers/plans/2026-05-09-compact-todo-interactive-memory-and-test-grooming.md)
  - [../../.agents/skills/compact-todo/SKILL.md](../../.agents/skills/compact-todo/SKILL.md)
  - [README.md](README.md)
  - [../todo.md](../todo.md)
  - [../log/index.md](../log/index.md)
  - [2026-05-09-1812-compact-todo-interactive-design.md](../log/2026-05-09-1812-compact-todo-interactive-design.md)
  - [2026-05-09-1834-compact-todo-implementation.md](../log/2026-05-09-1834-compact-todo-implementation.md)
  - [2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md](../log/2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md)

## Next Step

- One live compact session plus this follow-up hardening pass now cover protected-subtree compaction and explicit tree/index preservation.
- Next pressure target is grouped stale-test review under `Go2Pvcnn/tests/`, since the live sessions so far only scanned the tree and did not ask for delete/rewrite decisions.

## Node Details

### T002a written spec review gate

- status: done
- why-created:
  - the conversation already approved the design direction
  - the written spec needed explicit review before implementation
- evidence:
  - [2026-05-09-1812-compact-todo-interactive-design.md](../log/2026-05-09-1812-compact-todo-interactive-design.md)
- outcome:
  - the user approved the design content and asked to proceed directly with implementation

### T002b live usage pressure verification

- status: verify
- why-created:
  - the first implementation pass rewrote the skill content, but has only been verified by readback and grep
  - the new behavior still needs a real compact session to confirm the conversation flow feels right
- evidence:
  - [2026-05-09-1834-compact-todo-implementation.md](../log/2026-05-09-1834-compact-todo-implementation.md)
  - [2026-05-09-1848-t100-branch-compact-except-t116.md](../log/2026-05-09-1848-t100-branch-compact-except-t116.md)
- acceptance:
  - the skill asks for future direction first
  - the skill groups destructive note decisions instead of interrupting per node
  - the skill scans `Go2Pvcnn/tests/` and groups stale-test decisions by module family and candidate type
  - the skill asks the user when relevance is ambiguous
  - child compaction preserves visible parent/child tree and Obsidian index paths instead of flattening the subtree into prose
- live-session result:
  - the session safely preserved the active `T116` subtree while compacting sibling historical nodes into a linked context page
  - no destructive archive/delete/merge action was needed for this branch compact pass
  - the remaining unpressured behavior is grouped stale-test decision handling

### T002c tree-preserving child compaction and Obsidian index hardening

- status: done
- why-created:
  - the current skill described shrinking parent pages and moving child material, but it did not explicitly forbid flattening the tree during compaction
  - the user requested Obsidian-friendly index preservation when compacting child nodes
- evidence:
  - [2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md](../log/2026-05-10-2139-compact-todo-tree-obsidian-index-hardening.md)
- outcome:
  - `compact-todo` now requires visible parent/child index paths, named child entries, and no orphan child pages during compaction
  - [README.md](README.md) now states the same navigation contract for repository memory
