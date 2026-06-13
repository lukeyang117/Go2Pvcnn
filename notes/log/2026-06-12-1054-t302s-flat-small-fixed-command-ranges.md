# T302s Flat-Small Fixed Command Ranges

- Purpose: fix the command-range mismatch exposed by run `2026-06-11_22-15-56`.
- Stage: RL config / flat-small curriculum progression.
- Related todo: [T302s](../todo/T302s-env-level-collision-curriculum-plan.md)
- Procedure:
  - Added a static cfg test requiring the flat-small training command range:

```text
lin_vel_x = (0.6, 1.0)
lin_vel_y = (-0.2, 0.2)
ang_vel_z = (-0.3, 0.3)
```

  - Watched RED fail because current `lin_vel_x` was `(-0.1, 0.1)`.
  - Set the ranges in `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg.__post_init__`.
  - Ran focused local tests, pycompile, and an `env_isaacsim` smoke.
- Input conditions:
  - Previous TensorBoard readout showed `mean_terrain_level` dropped to `0` because velocity curriculum was disabled while command ranges stayed tiny.
  - Terrain move-up threshold is `4m` for the `8m` flat tiles.
- Key metrics:
  - RED: `test_flat_small_avoidance_cfg_static_contract` failed with `(-0.1, 0.1) != (0.6, 1.0)`.
  - Targeted GREEN: `1 passed`.
  - Focused suite: `180 passed, 1 warning`.
  - `py_compile`: exit `0`.
  - Real smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

  - Real smoke result: exit `0`; generated run `2026-06-12_10-53-23`.
  - Saved `env_cfg.yaml` confirms `lin_vel_x=(0.6,1.0)`, `lin_vel_y=(-0.2,0.2)`, `ang_vel_z=(-0.3,0.3)`.
- Result:
  - Flat-small no longer relies on the removed velocity curriculum to expand out of the tiny initial command range.
- Conclusion:
  - The next training run should have enough commanded displacement to satisfy the terrain curriculum distance gate when episodes are stable.
- Follow-up:
  - Run a short resumed training/TensorBoard check and verify `mean_terrain_level` no longer collapses to zero.
- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
