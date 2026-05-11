# compact-todo non-T116 subtree compression

## Purpose

- Compress repository memory while preserving the active `T116/T117` subtree untouched.
- Reduce detail in completed non-`T116` branch pages and tighten dashboard/log index focus around final T116 authority and current cleanup work.

## Stage

- notes workflow / branch and index compact

## Related Todo

- [T002/T002b](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002b-live-usage-pressure-verification)

## Command / Procedure

- Read repository constraints and memory entrypoints:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [../todo/T100-pre-t116-history.md](../todo/T100-pre-t116-history.md)
  - [../todo/T117-together-planner-test-and-todo-cleanup.md](../todo/T117-together-planner-test-and-todo-cleanup.md)
  - [index.md](index.md)
- Treated `T116` and `T117` as protected active surfaces.
- Compressed non-`T116` content only:
  - [../todo/T001-inspire-skill-design.md](../todo/T001-inspire-skill-design.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
- Confirmed that `T100` historical detail already lives in [../todo/T100-pre-t116-history.md](../todo/T100-pre-t116-history.md), so no destructive move was needed there.
- Did not archive, delete, or merge any protected `T116` subtree content.

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Baseline ref: `130c635`
- Current work ref: `working tree after T116h/T117 with unrelated planner dirt present`

## Key Metrics

- Protected active subtrees left untouched: `2`
  - `T116`
  - `T117`
- Non-`T116` branch pages compressed: `2`
  - `T001`
  - `T200`
- Dashboard/log surfaces tightened: `2`
  - `notes/todo.md`
  - `notes/log/index.md`
- Destructive actions executed: `0`

## Result

- Pass

## Conclusion

- Non-`T116` memory is now shorter and more navigable.
- `T116/T117` remain the only active together-planner surfaces in the main branch page and dashboard.
- Historical detail remains reachable through dedicated history pages and logs rather than the active root/branch surfaces.

## Follow-up

- Keep using [../todo/T100-pre-t116-history.md](../todo/T100-pre-t116-history.md) for pre-`T116` evidence instead of re-expanding the active `T100` page.
- If future compact sessions need more reduction, target `notes/log/index.md` recency window before touching protected active branches.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: `working tree on top of 130c635 (2026-05-10 18:43 +0800); compacted non-T116 subtree surfaces`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T001-inspire-skill-design.md](../todo/T001-inspire-skill-design.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
