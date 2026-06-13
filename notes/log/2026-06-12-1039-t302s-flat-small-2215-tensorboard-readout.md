# T302s Flat-Small 22:15 TensorBoard Readout

- Purpose: inspect run `2026-06-11_22-15-56` after env-level collision curriculum and clearance reward scale changes.
- Stage: training metrics / curriculum and semantic reward diagnostics.
- Related todo: [T302s](../todo/T302s-env-level-collision-curriculum-plan.md) / [T302r](../todo/T302r-go2-geometry-clearance-reward-plan.md)
- Procedure:
  - Parsed TensorBoard event file with `env_isaacsim` TensorBoard `EventAccumulator`.
  - Inspected saved `env_cfg.yaml`.
  - Re-read IsaacLab `ManagerBasedRLEnv._reset_idx()` and `TerrainImporter.update_env_origins()`.
- Input conditions:
  - Run dir: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_22-15-56`
  - Model range: `model_20000.pt` through `model_27200.pt`.
- Key metrics:
  - Scalar tags: `36`; curriculum tags now include only `Curriculum/terrain_levels/mean_terrain_level`.
  - `mean_terrain_level`: first `0.4938`, max `0.4938`, last `0.0`, last-100 mean `0.0`; nonzero `171/7209`.
  - Terrain level changes were almost all early downward changes; it reached `0.0` by step `20170` and stayed there through step `27207`.
  - `semantic_body_part_clearance`: nonzero `3890/7209`; min `-0.01010`; mean `-2.23e-4`; last-100 mean `-1.25e-4`; last-20 nonzero `12/20`.
  - `semantic_contact_collision`: nonzero `81/7209`; last-100 mean `0.0`.
  - `Train/mean_reward`: last-100 mean `38.04`.
  - `Train/mean_episode_length`: last-100 mean `1000.0`.
  - `track_lin_vel_xy`: last-100 mean `1.4738`.
  - `base_contact`: nonzero `2/7209`; `bad_orientation`: nonzero `198/7209`, last-100 mean `0.0`.
- Config finding:
  - Saved `env_cfg.yaml` shows flat-small training command ranges are still:

```text
lin_vel_x = (-0.1, 0.1)
lin_vel_y = (-0.1, 0.1)
```

  - `lin_vel_cmd_levels: null`, so the velocity curriculum is disabled but the command range did not get set to `limit_ranges`.
  - Terrain size is `(8.0, 8.0)`, so the terrain move-up distance threshold is `4.0m`.
- Result:
  - The new curriculum metric cleanup worked: old noisy curriculum scalars are gone.
  - The clearance reward scale worked in the sense that the signal is now visible and nonzero in about `54%` of scalar samples.
  - The terrain curriculum did not progress. It started around mean level `0.49`, then rapidly fell to `0` and stayed there.
- Root-cause hypothesis:
  - The dominant blocker is not the global semantic gate anymore. The saved config still uses very small command speeds while the terrain move-up threshold remains `4m`. With `lin_vel_x/y` at only `[-0.1, 0.1]`, the policy cannot reliably satisfy `distance > terrain_size / 2` during a 20s episode, so reset-time terrain updates mostly downgrade or stay at zero.
  - This is consistent with `mean_episode_length=1000`: locomotion is stable, but stable low-speed walking does not satisfy the distance-based curriculum upgrade rule.
- Follow-up:
  - Fix flat-small training command range after removing velocity curriculum: set `commands.base_velocity.ranges = commands.base_velocity.limit_ranges` or use an avoidance-specific forward-biased command range.
  - Consider relaxing terrain move-up threshold for flat-small after the command-range fix if `mean_terrain_level` still stays at zero.
- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
