# compact-todo tree and Obsidian index hardening

## Purpose

- Fix the `compact-todo` skill so child-node compaction preserves the todo tree instead of flattening it.
- Make Obsidian index/navigation surfaces explicit for the root dashboard, branch pages, and compacted child pages.

## Stage

- repository tooling / local skill hardening

## Related Todo

- [T002/T002c](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002c-tree-preserving-child-compaction-and-obsidian-index-hardening)

## Command / Procedure

- Read repository constraints and memory entrypoints:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
  - [../todo.md](../todo.md)
  - [../todo/README.md](../todo/README.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [index.md](index.md)
- Read the current skill and related design/memory records:
  - [../../.agents/skills/compact-todo/SKILL.md](../../.agents/skills/compact-todo/SKILL.md)
  - [../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md](../../docs/superpowers/specs/2026-05-09-compact-todo-interactive-memory-and-test-grooming-design.md)
  - [2026-05-09-1834-compact-todo-implementation.md](2026-05-09-1834-compact-todo-implementation.md)
  - [2026-05-10-1843-compact-todo-non-t116-subtree-compression.md](2026-05-10-1843-compact-todo-non-t116-subtree-compression.md)
- Hardened these contracts:
  - [../../.agents/skills/compact-todo/SKILL.md](../../.agents/skills/compact-todo/SKILL.md)
  - [../todo/README.md](../todo/README.md)
  - [../todo.md](../todo.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [index.md](index.md)
- Added explicit rules for:
  - visible parent -> child tree preservation during child compaction
  - Obsidian index surfaces for the root dashboard and branch pages
  - upward/downward link paths between parent pages, child pages, archives, and log chains
  - validation against orphan child pages and flattening shortcuts

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Baseline ref: `979b2b5`
- Current work ref: `working tree on top of 979b2b5 (2026-05-10 21:39 +0800); compact-todo tree/index hardening with unrelated planner/notes dirt present`

## Key Metrics

- Skill contract sections hardened: `5`
- Navigation contract files aligned: `5`
  - `.agents/skills/compact-todo/SKILL.md`
  - `notes/todo/README.md`
  - `notes/todo.md`
  - `notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md`
  - `notes/log/index.md`
- New compact log files: `1`
- Verification checks: `4`

## Result

- Pass

## Conclusion

- `compact-todo` now treats child compaction as tree-preserving reshaping instead of flattening.
- The root dashboard and branch pages are explicitly treated as Obsidian index surfaces, and validation now checks for visible parent/child link paths and orphan pages.
- The remaining live-usage gap under `T002b` is still grouped stale-test decision batching across `Go2Pvcnn/tests/`.

## Follow-up

- Pressure-test grouped stale-test review again now that tree/index preservation is explicit.
- If future compact sessions create deeper child pages, verify the parent anchor/backlink pattern in a real branch split.

## Git Refs

- Baseline Ref: `979b2b5`
- Candidate Ref: `working tree on top of 979b2b5 (2026-05-10 21:39 +0800); compact-todo tree/index hardening with unrelated planner/notes dirt present`
- Key Files:
  - [../../.agents/skills/compact-todo/SKILL.md](../../.agents/skills/compact-todo/SKILL.md)
  - [../todo/README.md](../todo/README.md)
  - [../todo.md](../todo.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [index.md](index.md)
