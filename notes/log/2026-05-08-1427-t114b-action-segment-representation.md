# T114b Action-Segment Representation

## Meta

- Time: `2026-05-08 14:27 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T114b](../todo/T100-batched-together-planner-gpu-migration.md#t114b-candidate-action-segment-representation)

## Purpose

- Verify the second execution leaf under `T114`: candidate representation is upgraded from endpoint-like touchdown data to explicit action-segment diagnostics while preserving fixed `K=3`.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114b_'`
  - result: `2 failed, 21 deselected`
  - failing reason: missing `candidate_state_tag` / `candidate_anchor_references` diagnostics on `TogetherRollout`
- GREEN after minimal implementation:
  - the same `t114b_` subset rerun passed

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114b_'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
5. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Key Metrics

- TDD red subset: `2 failed`
- focused `T114b` subset after implementation: `2 passed`
- semantic affected-union rerun: `23 passed`
- core affected-union rerun: `6 passed`
- guardrail rerun: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `candidate_action_segment_diagnostics_present` is now directly testable
- candidate-stage diagnostics now prove the rollout carries action-segment semantics rather than only endpoint touchdown data:
  - `candidate_state_tag`
  - `candidate_anchor_references`
  - `candidate_touchdown_targets`
  - `candidate_path_progress`
  - `candidate_pair_summary`
  - `candidate_posture_summary`
- the representation still preserves fixed `K=3`
- production hot-path changes remained tensorized and guardrail-clean

## Caveats

- These diagnostics live on `TogetherRollout` only for now, not yet on `TogetherPlannerResult`; that is consistent with `T114b` staying focused on front-end representation rather than later result-surface work.
- `candidate_pair_summary` and `candidate_posture_summary` are currently structural summaries; later leaves still need to give them stronger scoring/validation semantics.

## Conclusion

- `T114b` is complete and verified with TDD evidence plus final affected-union reruns on the final code state.
- Recommended next leaf is `T114c`.

## Git Refs

- Baseline Ref: `current working tree after T114a verification`
- Candidate Ref: `working tree with T114b action-segment representation changes and focused verification; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
