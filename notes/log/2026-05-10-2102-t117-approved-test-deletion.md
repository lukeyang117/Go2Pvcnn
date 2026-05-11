# T117 approved test deletion

## Purpose

- Apply the user-approved destructive cleanup for together-planner-adjacent tests.
- Keep only the current authority test surfaces and remove the rest of `Go2Pvcnn/tests/`.

## Stage

- notes/test grooming with explicit user-approved deletion

## Related Todo

- [T100/T117](../todo/T117-together-planner-test-and-todo-cleanup.md#t117c-remove-non-mainline-parity-and-historical-traceability-surfaces)

## Command / Procedure

- Reconfirmed current authority surfaces before deletion:
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/fixtures/__init__.py](../../Go2Pvcnn/tests/fixtures/__init__.py)
- Deleted all other test and benchmark files under `Go2Pvcnn/tests/` after explicit user approval.
- Kept note links and traceability by updating todo/log memory instead of leaving the deletion implicit.

## Input Conditions

- User decision: delete everything in `Go2Pvcnn/tests/` except the current keep surfaces.
- Non-goal: no new replacement tests were written in this pass.

## Key Metrics

- Kept authority test files: `3`
- Kept fixture helper files: `2`
- Deleted test/benchmark/helper files: `37`

## Result

- Pass

## Conclusion

- The together-planner test tree is now intentionally reduced to the current authority surfaces only.
- Future test rebuilding can proceed from a much smaller base without older parity, historical traceability, or transitional runtime helper contracts.

## Follow-Up

- Rewrite new mainline tests only when they directly protect the post-`T116h` future direction.
- If broader non-together test coverage is needed later, reintroduce it intentionally rather than restoring the old mixed tree.

## Git Refs

- Baseline Ref: `working tree after T117 scan`
- Candidate Ref: `working tree on 2026-05-10 21:02 +0800; user-approved large test deletion`
- Key Files:
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/fixtures/__init__.py](../../Go2Pvcnn/tests/fixtures/__init__.py)
