# T113d Swing And Collision Model

## Meta

- Time: `2026-05-07 21:22 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T113d](../todo/T100-batched-together-planner-gpu-migration.md#t113d-height-aware-swing-and-continuous-collision-model)

## Purpose

- Verify the fourth execution leaf under `T113`: height-aware swing clearance plus continuous body/thigh/calf collision coverage under pure-GPU fixed-shape constraints.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
- [../../Go2Pvcnn/extension/batched_together_planner/kinematics.py](../../Go2Pvcnn/extension/batched_together_planner/kinematics.py)
- [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
- [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## Subagent Result Summary

- Swing clearance now queries batched height maxima along the airborne segment rather than relying only on the fixed global arc.
- `kinematics.py` now exposes tensorized world-frame hip/knee/foot keypoints for each leg.
- `costs.py` now computes separate continuous clearance penalties:
  - `J_collision_body`
  - `J_collision_leg`
- Whole-horizon minimum clearances are now surfaced:
  - `body_min_clearance`
  - `leg_min_clearance`
- Hard infeasible rejection now includes deep body/leg penetration while keeping mild close approach as soft penalty.

## Verification Commands

1. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/kinematics.py Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`

## Key Metrics

- focused `core + semantic planner` suite: `20 passed`
- together guardrail subset: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `body_min_clearance` and `leg_min_clearance` computed over whole horizon
  - covered through new result fields and focused tests
- `F6_body_collision_only`
  - covered by `test_body_collision_only_fixture_becomes_infeasible`
- `F7_leg_collision_only`
  - covered by `test_leg_collision_only_fixture_becomes_infeasible`
- `F9_mild_clearance_penalty_but_feasible`
  - covered by `test_mild_clearance_penalty_can_increase_while_remaining_feasible`
- planner hot path remains pure GPU and fixed shape
  - guardrail suite stayed green after the changes

## Caveats

- Broader repository `pytest` remains blocked by the known `Go2Pvcnn/tests/conftest.py` dependency on missing `scripts.go2fp`; this leaf stayed on focused together-only verification.
- The collision model remains a merged-height fixed-template approximation, so it covers height-representable terrain/obstacle risk but not side-overhang or other non-height-representable geometry beyond the approved design contract.
- This leaf added tuning knobs in `config.py`, which is expected by the design but should still be reviewed during `T113e` acceptance surfacing.

## Conclusion

- `T113d` is complete and verified within the current together-only test path.
- The final recommended execution step is `T113e`, which should close the remaining diagnostics-facing metric and fixture traceability work.

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 (2026-05-07 21:22 +0800); T113d code/test changes plus notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/kinematics.py](../../Go2Pvcnn/extension/batched_together_planner/kinematics.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
