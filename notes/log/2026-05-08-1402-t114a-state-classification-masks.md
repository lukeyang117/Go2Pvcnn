# T114a State Classification Masks

## Meta

- Time: `2026-05-08 14:02 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T114a](../todo/T100-batched-together-planner-gpu-migration.md#t114a-state-classification-masks-and-corridor-summaries)

## Purpose

- Verify the first execution leaf under `T114`: unified pure-GPU state classification and corridor summaries for `cruise / approach / ready_to_cross / front_cross / rear_follow / bypass / clear`.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114a_'`
  - result: `7 failed, 14 deselected`
  - failing reason: new tests expected `TogetherRollout.state_code` and related corridor diagnostics that did not yet exist
- GREEN after minimal implementation:
  - the same `t114a_` subset rerun passed

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114a_'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
5. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Key Metrics

- TDD red subset: `7 failed`
- focused `T114a` subset after implementation: `7 passed`
- semantic affected-union rerun: `21 passed`
- core affected-union rerun: `6 passed`
- guardrail rerun: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- deterministic fixtures now explicitly cover:
  - `cruise`
  - `approach`
  - `ready_to_cross`
  - `front_cross`
  - `rear_follow`
  - `bypass`
  - `clear`
- no-small scenes remain in the same unified framework and classify as `cruise`
- implementation remains pure GPU and fixed shape:
  - no `for`
  - no `numpy`
  - no CPU sync / host branching added in production hot-path code
  - guardrail suite passed on the final code state

## Caveats

- This leaf intentionally keeps the new diagnostics on `TogetherRollout`, not yet on `TogetherPlannerResult`; surfacing those candidate-stage diagnostics further is deferred to later `T114` leaves.
- Thresholds are still heuristic at this stage and are expected to be refined by later leaves once margin, pair-consistency, and path-clearance scoring are added.

## Conclusion

- `T114a` is complete and verified with TDD evidence plus final affected-union reruns on the final code state.
- Recommended next leaf is `T114b`.

## Git Refs

- Baseline Ref: `current working tree after T114 todo mapping`
- Candidate Ref: `working tree with T114a state-mask changes and focused verification; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
