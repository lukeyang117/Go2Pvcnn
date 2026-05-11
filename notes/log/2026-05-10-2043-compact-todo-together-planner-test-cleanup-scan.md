# compact-todo together-planner test cleanup scan

## Purpose

- Run a `compact-todo` state scan after `T116g` and `T116h` completed.
- Identify which together-planner notes and tests still reflect stale pre-final-state assumptions.
- Prepare grouped keep/rewrite/ambiguous decisions without deleting or rewriting tests yet.

## Stage

- notes/test grooming scan

## Related Todo

- [T002/T002b](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md#t002b-live-usage-pressure-verification)
- [T100/T117](../todo/T100-batched-together-planner-gpu-migration.md#t117-together-planner-test-and-todo-cleanup-after-t116h)

## Command / Procedure

- Read repository constraints and planner pre-read notes:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
  - [../todo/README.md](../todo/README.md)
  - [../human/human-08-extension-planner-reading-guide.md](../human/human-08-extension-planner-reading-guide.md)
  - [../human/human-09-extension-planner-mapping.md](../human/human-09-extension-planner-mapping.md)
- Read active/final T116 evidence:
  - [2026-05-10-1953-t116g-env-isaacsim-runtime-diagnostics.md](2026-05-10-1953-t116g-env-isaacsim-runtime-diagnostics.md)
  - [2026-05-10-2017-t116h-final-review-authority.md](2026-05-10-2017-t116h-final-review-authority.md)
  - [2026-05-09-2150-t116f-deterministic-guardrail-cleanup.md](2026-05-09-2150-t116f-deterministic-guardrail-cleanup.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
- Scanned the full `Go2Pvcnn/tests/` tree and focused on together-planner families plus related runtime/benchmark files.
- Grepped for stale `35`-frame, `K=3`, `front_cross/rear_follow/clear`, and old runtime-schema literals.

## Input Conditions

- Direction signal: keep the future mainline centered on `T116g/T116h` final together-planner authority and clean stale together-planner tests/todo surfaces.
- Non-goal: no destructive note archive/delete, no test deletion, no code-path rewrites in this scan pass.

## Key Metrics

- Root/dashboard drift findings: `4`
- Together-planner rewrite-candidate test bundles: `4`
- Ambiguous historical/parity bundles: `2`
- Explicit keep bundles: `3`
- Destructive actions performed: `0`

## Result

- Scan recorded

## Findings

- Notes drift:
  - [../todo.md](../todo.md) duplicated `T002b` in both `Active Fronts` and `Open Leaves`.
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md) still described `T116g` as doing and still pointed `Next Step` at pre-final-state execution.
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md) still said live pressure testing was pending even though one compact session already passed.
  - [index.md](index.md) had the recent `T116g/T116h` rows but the topic index had not yet absorbed them.
- Test grouping:
  - `keep`:
    - `Go2Pvcnn/tests/test_batched_together_core.py`
    - `Go2Pvcnn/tests/test_batched_together_guardrails.py`
    - `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
  - `rewrite-candidate`:
    - `Go2Pvcnn/tests/test_batched_together_runtime_path.py`
    - `Go2Pvcnn/tests/test_batched_together_manager.py`
    - `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`
    - `Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py`
  - `ambiguous`:
    - `Go2Pvcnn/tests/test_batched_together_parity.py`
    - historical traceability blocks inside `Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Conclusion

- The mainline implementation/testing story is already closed by `T116h`, but memory surfaces and several together-planner test helpers still describe older or transitional states.
- The highest-signal next cleanup is to rewrite hardcoded `35`-frame together-runtime/manager/benchmark helpers and then decide what to do with parity/historical traceability bundles.

## Follow-Up

- Update notes so `T116g/T116h` are recorded as closed and `T117` becomes the active cleanup leaf.
- Present grouped test decisions before deleting or downgrading any ambiguous historical/parity surface.

## Git Refs

- Baseline Ref: `working tree with T116h already complete`
- Candidate Ref: `working tree on 2026-05-10 20:43 +0800; notes-only compact scan for together-planner test cleanup`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T002-compact-todo-interactive-memory-and-test-grooming.md](../todo/T002-compact-todo-interactive-memory-and-test-grooming.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
  - [../../Go2Pvcnn/tests/test_batched_together_runtime_path.py](../../Go2Pvcnn/tests/test_batched_together_runtime_path.py)
  - [../../Go2Pvcnn/tests/test_batched_together_manager.py](../../Go2Pvcnn/tests/test_batched_together_manager.py)
  - [../../Go2Pvcnn/tests/test_batched_together_parity.py](../../Go2Pvcnn/tests/test_batched_together_parity.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
