# T302q Flat-Small Remove Velocity Curriculum

## Purpose

Remove the velocity-command curriculum from only the flat-small avoidance training config because the policy already learned the speed range.

## Stage

RL config / curriculum wiring.

## Related Todo

- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)
- Related geometry branch: [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)

## Command / Procedure

Changed `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg.__post_init__()` so:

```python
self.curriculum.lin_vel_cmd_levels = None
```

This keeps `terrain_levels` active and leaves the base `TeacherElevationTrajectoryMpcSemanticEnvCfg` velocity curriculum unchanged.

Verification:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
python -m py_compile Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance --num_envs 8 --max_iterations 1 --headless
```

## Input Conditions

- User requested deleting the speed curriculum in this config only.
- Flat-small still needs terrain/semantic curriculum behavior.

## Key Metrics

- RED: static contract failed because `cfg.curriculum.lin_vel_cmd_levels` was still set.
- GREEN targeted static: `1 passed`.
- Backend + reward focused: `158 passed, 1 warning`.
- Semantic curriculum focused: `20 passed`.
- `py_compile`: exit `0`.
- Real IsaacLab smoke: exit `0`.
- Real Curriculum Manager active terms: `1`, only `terrain_levels`.

## Result

Pass. Flat-small avoidance no longer mounts `lin_vel_cmd_levels`; the base config still does.

## Conclusion

The change is scoped to `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg` and does not remove velocity commands or tracking rewards. It only stops expanding velocity command limits through the curriculum term.

## Follow-Up

- Continue T302r larger nonzero-rate/perf/TensorBoard checks with the speed curriculum disabled.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree on 2026-06-11 16:21
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
