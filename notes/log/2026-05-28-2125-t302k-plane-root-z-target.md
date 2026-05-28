# 2026-05-28 21:25 T302k Plane Root Z Target

## Purpose

Add a plane-only root z target loss using `MpcPlannerTerrain.is_plane_terrain`, with default target height from the initial root z.

## Stage

`extension/batch_mpc_planner` sampled parametric losses.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'plane_root_z_target'
```

Initial failure:

```text
ImportError: cannot import name 'parametric_plane_root_z_target_loss'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'plane_root_z_target'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or semantic'
```

## Input Conditions

- Baseline ref: `1a15677`.
- `is_plane_terrain=None` should disable the loss.
- `target_height_m=None` uses current state root z.

## Key Metrics

- Focused plane root-z test: `1 passed, 117 deselected`.
- Backend parametric/semantic subset: `32 passed, 86 deselected, 1 warning`.
- Pycompile: pass.

## Result

Pass locally.

## Conclusion

Added `parametric_plane_root_z_target_loss()` and sampled key `parametric_plane_root_z_target`, gated by `terrain.is_plane_terrain`.

## Follow-Up

Continue with Task 9 plane low-small FK semantic collision probe.

## Git Refs

- Baseline Ref: `1a15677`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
