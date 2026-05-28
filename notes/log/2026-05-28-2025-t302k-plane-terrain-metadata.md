# 2026-05-28 20:25 T302k Plane Terrain Metadata

## Purpose

Carry automatic plane-terrain metadata into `MpcPlannerTerrain` so the later plane-only root-z loss can gate from IsaacLab terrain construction instead of manual `terrain_col` configuration.

## Stage

`extension/batch_mpc_planner` terrain contract and IsaacLab MPC manager terrain boundary.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red metadata tests:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'is_plane_terrain'
```

Initial failures:

```text
TypeError: MpcPlannerTerrain.__init__() got an unexpected keyword argument 'is_plane_terrain'
TypeError: build_mpc_terrain_from_scanner() got an unexpected keyword argument 'is_plane_terrain'
```

Manager red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'plane_terrain_metadata'
```

Initial manager failure:

```text
terrain.is_plane_terrain is None
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'is_plane_terrain or build_mpc_terrain or plane_terrain_metadata'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/terrain.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/batch_mpc_planner/planner.py
```

## Input Conditions

- Baseline ref: `faaf61e`.
- Task 2 requires plane-only root-z metadata without using row/difficulty or manual terrain column config.

## Key Metrics

- Focused backend metadata tests: `4 passed, 105 deselected`.
- Pycompile: pass.

## Result

Pass locally.

## Conclusion

`MpcPlannerTerrain` now carries optional `is_plane_terrain`. Scanner construction, terrain subset, planner normal/subset, and `MpcTrajectoryManager` preserve it. The manager infers plane masks from IsaacLab `terrain_types` and `terrain_generator.sub_terrains` names, with only `flat` and `plane` treated as plane terrain.

## Follow-Up

Continue with Task 3 GPU low-small circle approximation.

## Git Refs

- Baseline Ref: `faaf61e`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/types.py](../../Go2Pvcnn/extension/batch_mpc_planner/types.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
