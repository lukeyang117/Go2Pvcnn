# Flat-Small 2026-06-14 00-22-58 TensorBoard Readout

## Purpose

Diagnose why the flat-small run has low tracking reward, no curriculum progress, and short trajectories.

## Stage

Training metrics / flat-small avoidance / MPC runtime configuration.

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Run

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-14_00-22-58
```

Process still running at readout:

```text
python Go2Pvcnn/scripts/train.py ... --num_envs 1024 ... --mpc_num_envs 1
```

## Key Scalars

Steps:

```text
19999 -> 22989
```

Tracking:

```text
Episode_Reward/track_lin_vel_xy last100 = 0.0005269
Episode_Reward/track_ang_vel_z last100 = 0.003201
Metrics/base_velocity/error_vel_xy last100 = 0.001894
Metrics/base_velocity/error_vel_yaw last100 = 0.0008895
```

Curriculum:

```text
Curriculum/terrain_levels/mean_terrain_level first = 0.4643
Curriculum/terrain_levels/mean_terrain_level last100 = 0.0
nonzero points = 14 / 2991
```

Episode length / terminations:

```text
Train/mean_episode_length first100 = 28.58
Train/mean_episode_length last100 = 8.25
Episode_Termination/time_out last100 = 0.00075
Episode_Termination/base_contact last100 = 0.00050
Episode_Termination/bad_orientation last100 = 124.04
```

Semantic / reference:

```text
Episode_Reward/semantic_body_part_clearance last100 = -0.001269
Episode_Reward/semantic_foot_over_clearance last100 = 0.0
Episode_Reward/reference_foot_pos last100 = 3.37e-7
```

Saved cfg:

```text
scene.num_envs = 1024
mpc_planner_cfg.runtime.parallel_plan_batch_size = 1
GoalAnchoredVelocityCommand:
  vx_abs_range = (0.6, 1.0)
  vy_abs_range = (0.6, 1.0)
  yaw_range = (-0.8, 0.8)
```

## Conclusion

The main failure is not simply low tracking reward. Episodes are collapsing almost immediately: `mean_episode_length` is around `8` steps and `bad_orientation` dominates terminations. Because flat-small curriculum only moves up on successful timeout episodes, curriculum cannot progress and falls back to terrain level `0`.

The run also used `--mpc_num_envs 1`, so only one environment is sampled by MPC per replan. That explains the nearly absent `reference_foot_pos` reward and the user's observation that trajectories are very short/weak.

The likely combined issue is:

1. `--mpc_num_envs 1` starves the MPC reference path.
2. Goal-anchored command is too aggressive for the loaded old checkpoint: body-frame x/y magnitudes are both `0.6-1.0`, so diagonal speed can be about `0.85-1.41 m/s`, plus yaw up to `0.8 rad/s`.
3. The policy falls by bad orientation before it can collect useful tracking/semantic/curriculum signal.

## Recommendation

Stop this run. Restart with at least:

```text
--mpc_num_envs 64
```

If bad-orientation remains high after a short sanity window, reduce the command shock for warm start before pushing full all-direction speed.
