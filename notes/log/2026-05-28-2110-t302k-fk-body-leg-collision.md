# 2026-05-28 21:10 T302k FK Body Leg Collision

## Purpose

Add FK body/leg terrain collision checks for realized geometry: foot, knee, shank samples, root, and underbody samples.

## Stage

`extension/batch_mpc_planner` FK geometry and final parametric loss breakdown.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'fk_body_leg_collision or shank_pos_world_alias'
```

Initial failure:

```text
ImportError: cannot import name 'FkCollisionMargins'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'fk_body_leg_collision or shank'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/kinematics.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/config.py
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or semantic'
```

## Input Conditions

- Baseline ref: `a5b5772`.
- Confirmed parameters: `fk_foot_clearance_margin_m`, `fk_knee_clearance_margin_m`, `fk_shank_clearance_margin_m`, `fk_root_clearance_margin_m`, `fk_underbody_clearance_margin_m`, `fk_shank_sample_count`, `fk_underbody_sample_count`.

## Key Metrics

- Focused FK collision tests: `5 passed, 111 deselected`.
- Backend parametric/semantic subset: `32 passed, 84 deselected, 1 warning`.
- Pycompile: pass.

## Result

Pass locally.

## Conclusion

`MpcLegPoints` now exposes `shank_pos_world` as an alias for shank samples. Added `parametric_fk_body_leg_collision_loss()` and final loss key `parametric_fk_body_leg_collision`.

Important limitation: the loss is computed after optimizing and solving joint/FK output, then added to final `loss_breakdown/cost_breakdown`. It is not currently part of the Adam inner loop because that loop decodes target trajectory before the post-solve FK pass.

## Follow-Up

Continue with Task 7 optimized-vs-FK trajectory consistency.

## Git Refs

- Baseline Ref: `a5b5772`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
