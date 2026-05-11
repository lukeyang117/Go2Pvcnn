# T113b Always-On K=3 Candidate Axis

## Meta

- Time: `2026-05-07 20:40 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T113b](../todo/T100-batched-together-planner-gpu-migration.md#t113b-always-on-k3-foothold-policy-candidate-axis)

## Purpose

- Verify the second execution leaf under `T113`: fixed always-on `K=3` candidate expansion with foothold-level differentiation rather than route-offset-only behavior.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
- [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## Subagent Result Summary

- Planner candidate expansion is now always fixed at `K=3`, including height-only scenes.
- Candidate differentiation moved into touchdown/support generation through candidate-specific body-frame lateral foothold bias before final selection.
- Core and semantic planner tests now explicitly cover:
  - `candidate_count == 3` in obstacle-free, small, and large scenes
  - obstacle-free uneven terrain selecting a non-center candidate when support quality is better
  - rollout-level candidate touchdown body-frame offsets differing before final selection

## Verification Commands

1. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`

## Key Metrics

- focused `core + semantic planner` suite: `13 passed`
- together guardrail subset: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `candidate_count == 3` in obstacle-free, small, and large fixtures
  - covered by updated core + semantic planner tests
- obstacle-free uneven terrain may choose a non-center candidate
  - covered by `test_height_only_uneven_terrain_may_choose_non_center_candidate`
- candidate diagnostics reflect real foothold-policy differences rather than only later route offsets
  - covered by `test_candidate_touchdown_body_frame_offsets_change_with_candidate_policy`

## Caveats

- Full repository `pytest` remains blocked by the existing `Go2Pvcnn/tests/conftest.py` import of missing `scripts.go2fp`; this leaf used the same together-only `--noconftest` verification path as other focused planner leaves.
- The public diagnostics surface is still `selected_route_offset` and `semantic_candidate_costs`; this leaf proves foothold-policy differentiation through deterministic rollout tests rather than by adding a new per-candidate touchdown tensor to the result schema.

## Conclusion

- `T113b` is complete and verified within the current together-only test path.
- The next recommended execution step is `T113c`, where the always-on candidate axis gains explicit `small` crossing preference and `large` bypass / forward-refusal semantics.

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 (2026-05-07 20:40 +0800); T113b code/test changes plus notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
