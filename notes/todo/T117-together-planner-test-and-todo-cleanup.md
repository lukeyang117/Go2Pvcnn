# T117 Together Planner Test And Todo Cleanup

## Current State

- `T116` is complete through `T116h`, so this node owns only post-completion memory/test grooming.
- The active goal is to remove stale together-planner test and todo surfaces that still describe `35`-frame, pre-final-state, or historical traceability contracts as if they were current authority.
- Current authority surfaces stay:
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
- User has now approved destructive deletion of all other `Go2Pvcnn/tests/` files.

## Open Children

- [T117a](#t117a-notes-and-index-alignment): compress root/T100/T002/log surfaces around final T116 state.
- [T117b](#t117b-runtimemanagerbenchmark-35-frame-cleanup): rewrite together runtime/manager/benchmark tests that still hardcode `35`.
- [T117c](#t117c-remove-non-mainline-parity-and-historical-traceability-surfaces): remove or relocate parity/historical-traceability surfaces that no longer serve the future mainline.

## Closed Children Archive

- none

## Related Logs

- [2026-05-10-2043-compact-todo-together-planner-test-cleanup-scan.md](../log/2026-05-10-2043-compact-todo-together-planner-test-cleanup-scan.md)
- [2026-05-10-2102-t117-approved-test-deletion.md](../log/2026-05-10-2102-t117-approved-test-deletion.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `T116h final authority on 2026-05-10 20:17 +0800`
- Current Work Ref: `working tree after T116h; test/todo cleanup in progress`
- Key Files:
  - [T100-batched-together-planner-gpu-migration.md](T100-batched-together-planner-gpu-migration.md)
  - [../todo.md](../todo.md)
  - [../log/index.md](../log/index.md)
  - [../../Go2Pvcnn/tests/test_batched_together_runtime_path.py](../../Go2Pvcnn/tests/test_batched_together_runtime_path.py)
  - [../../Go2Pvcnn/tests/test_batched_together_manager.py](../../Go2Pvcnn/tests/test_batched_together_manager.py)
  - [../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py](../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py)
  - [../../Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py](../../Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## Next Step

- Finish non-destructive notes alignment first.
- Then rewrite/remove non-keep test surfaces in grouped batches, preserving only current authority and useful historical notes.

## Node Details

### T117a notes and index alignment

- status: `doing`
- why-created:
  - dashboard and branch memory drifted after `T116g/T116h` completed
  - test cleanup needs a stable note surface before touching test files
- target:
  - root dashboard
  - T100 branch
  - T002 branch
  - log index

### T117b runtime/manager/benchmark `35`-frame cleanup

- status: `done`
- why-created:
  - several together-planner test helpers still hardcode `reference_trajectory_horizon=35` and `reference_replan_interval_steps=35`
- target files:
  - [../../Go2Pvcnn/tests/test_batched_together_runtime_path.py](../../Go2Pvcnn/tests/test_batched_together_runtime_path.py)
  - [../../Go2Pvcnn/tests/test_batched_together_manager.py](../../Go2Pvcnn/tests/test_batched_together_manager.py)
  - [../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py](../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py)
  - [../../Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py](../../Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py)

### T117c remove non-mainline parity and historical-traceability surfaces

- status: `done`
- why-created:
  - raw parity and T113/T114/T115 traceability are no longer future-mainline authority for together planner
- target files:
  - [../../Go2Pvcnn/tests/test_batched_together_parity.py](../../Go2Pvcnn/tests/test_batched_together_parity.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
- outcome:
  - user approved broader deletion, so these surfaces were removed together with the rest of the non-keep test tree instead of being split into a new historical test module
