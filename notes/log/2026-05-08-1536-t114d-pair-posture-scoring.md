# T114d Pair And Posture Scoring

## Meta

- Time: `2026-05-08 15:36 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T114d](../todo/T100-batched-together-planner-gpu-migration.md#t114d-four-leg-consistency-and-whole-body-posture-scoring)

## Purpose

- Verify the fourth execution leaf under `T114`: candidate-stage front-pair consistency, rear-pair follow consistency, and whole-body posture scoring.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
- [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114d_'`
  - result: `3 failed, 25 deselected`
  - failing reason: missing `front_pair_consistency`, `rear_pair_follow_consistency`, and `body_posture_score` on `TogetherRollout`
- GREEN after minimal implementation:
  - the same `t114d_` subset rerun passed

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114d_'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
5. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Key Metrics

- TDD red subset: `3 failed`
- focused `T114d` subset after implementation: `3 passed`
- semantic affected-union rerun: `28 passed`
- core affected-union rerun: `6 passed`
- guardrail rerun: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `front_pair_consistency`, `rear_pair_follow_consistency`, and `body_posture_score` are now directly test-observable
- deterministic bad whole-body posture rejection is covered at candidate stage
- split-touchdown fixtures now prove front-pair and rear-pair metrics can rise independently
- selected-candidate cost breakdown now preserves:
  - `J_pair_consistency`
  - `J_body_posture`
- guardrail suite remained green, preserving pure-GPU/no-`for` hot-path constraints

## Caveats

- This leaf required two additional directly-related production files beyond the minimal original scope:
  - `config.py` for posture/consistency threshold and weight knobs
  - `planner.py` so the selected-candidate breakdown preserves the new terms
- Thresholds remain heuristic tuning knobs and may need retuning once later path-clearance and state-aware selection leaves are in place.
- These diagnostics are strongest on `TogetherRollout`; they are not yet standalone top-level `TogetherPlannerResult` fields.

## Conclusion

- `T114d` is complete and verified with TDD evidence plus final affected-union reruns on the final code state.
- Recommended next leaf is `T114e`.

## Git Refs

- Baseline Ref: `current working tree after T114c verification`
- Candidate Ref: `working tree with T114d pair/posture scoring changes and focused verification; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
