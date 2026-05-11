# T100 branch compact except T116

## Purpose

- Pressure-test the new `compact-todo` workflow in a real repository memory session.
- Compact the `T100` branch so only `T116` remains expanded and active.
- Preserve all non-`T116` history without touching the currently active `T116` subtree.

## Stage

- notes workflow / branch compact

## Related Todo

- [T002/T002b](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002b-live-usage-pressure-verification)
- [T100](../todo/T100-batched-together-planner-gpu-migration.md)

## Command / Procedure

- Read repository constraints and memory entrypoints:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
  - [../todo/README.md](../todo/README.md)
- Read active branch memory and recent T116 logs:
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [2026-05-09-1718-t116-design-review-and-todo-cleanup.md](2026-05-09-1718-t116-design-review-and-todo-cleanup.md)
  - [2026-05-09-1744-t116-plan-and-subagent-dispatch.md](2026-05-09-1744-t116-plan-and-subagent-dispatch.md)
  - [2026-05-09-1641-k5-mode-first-cross-small-design.md](2026-05-09-1641-k5-mode-first-cross-small-design.md)
- Scanned the full `Go2Pvcnn/tests/` tree to satisfy the skill's whole-tree review requirement.
- Compacted the non-`T116` portion of `T100` by:
  - moving pre-`T116` branch detail into [../todo/T100-pre-t116-history.md](../todo/T100-pre-t116-history.md)
  - replacing the old expanded historical block in [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md) with a short history index
  - adding the new branch page to [../todo.md](../todo.md)
  - aligning `T002` and the log index to record this live compact session

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Protected active subtree: `T116` and its `T116a-h` leaves
- Non-goal: no edits to in-progress `T116` content, no archival/deletion/merge of notes, no test rewrites/deletions

## Key Metrics

- Branch page compacted: `1`
- New historical context pages: `1`
- Protected active subtree preserved: `T116`
- Full `Go2Pvcnn/tests/` tree scanned: `1` pass
- Destructive actions performed: `0`

## Result

- Pass

## Conclusion

- The session validated that `compact-todo` can safely compact a large branch around one protected active subtree.
- `T100` now keeps `T116` as the only expanded active branch, while pre-`T116` detail remains reachable from a dedicated historical context page.
- This was a non-destructive reshape only; no archive/delete/merge decisions were needed.

## Follow-up

- Future `compact-todo` pressure testing should target grouped stale-test decisions under `Go2Pvcnn/tests/`.
- If `T100` grows again after `T116`, continue using linked historical/context pages instead of re-expanding the main branch page.

## Git Refs

- Baseline Ref: `working tree on top of 7cf6c11 with active planner edits in progress`
- Candidate Ref: `working tree on top of 7cf6c11 (2026-05-09 18:48 +0800); notes-only branch compaction around active T116`
- Key Files:
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [../todo/T100-pre-t116-history.md](../todo/T100-pre-t116-history.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
