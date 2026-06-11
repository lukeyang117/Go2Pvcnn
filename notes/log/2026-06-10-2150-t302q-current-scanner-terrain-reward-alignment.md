# T302q Current Scanner Terrain Reward Alignment

## Purpose

Record the user-requested rollback of the reward-private scanner refresh/root-anchor cache design. The near-field avoidance reward now uses the current IsaacLab scanner semantic/elevation maps and the same MPC terrain query helper instead of storing `map_root_pos_w/map_root_yaw_w`.

## Stage

RL reward / flat-small near-field avoidance.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

RED:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

GREEN:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Compile:

```bash
python -m py_compile Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_batch_mpc_backend.py
```

Residual scan:

```bash
rg -n "map_root|anchor_cache|_update_semantic_map_anchor_cache|cached scanner|cached semantic|scanner-root|缓存地图|map_root_pos|map_root_yaw|scanner 不是每个|低频缓存|grid_origin_xy|grid_resolution" ...
```

## Input Conditions

- Working tree already contained T302q flat-small avoidance cfg/reward/curriculum changes.
- User explicitly requested deleting the refresh/root-anchor handling and using the current IsaacLab-returned elevation and semantic maps aligned with MPC.

## Key Metrics

- RED: stale API/cache behavior reproduced; helper rejected `current_root_pos_w`, and wrapper kept stale anchor after current root/body translation.
- GREEN: focused test command `9 passed in 1.63s`.
- Compile: exit `0`.
- Residual scan: no matches in active code/tests/docs set.

## Result

Pass locally. `semantic_body_part_clearance_reward` now builds current scanner terrain from `scanner.data.elevation_map`, `scanner.data.semantic_map`, `scanner.data.pos_w`, and `scanner.data.quat_w`, then queries `height_at()` / `semantic_at()` for current foot/shank/thigh points.

## Conclusion

The reward no longer owns scanner refresh timing or root-anchor cache state. Flat-small cfg no longer overrides `semantic_height_scanner.update_period` to `0.2`.

## Follow-Up

Rerun the 1024-env performance smoke after any further observation-side optimization, because this pass only verifies local behavior and compile correctness.

## Git Refs

- Baseline Ref: `working tree after initial T302q implementation and speed experiments`
- Candidate Ref: `working tree @ 2026-06-10 21:50 CST`
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)
