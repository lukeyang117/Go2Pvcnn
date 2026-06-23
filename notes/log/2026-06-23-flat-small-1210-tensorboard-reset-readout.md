# Flat-Small 12:10 TensorBoard Reset Readout

## Purpose

Read TensorBoard for `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10` and decide whether reset thresholds should be relaxed.

## Stage

- Training metrics / flat-small avoidance / reset diagnosis
- Related todo: [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Input Conditions

- Run path: [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10)
- Event file: `events.out.tfevents.1781669017.enine.1576822.0`
- Checkpoints up to `model_14700.pt`
- Saved `env_cfg.yaml` has `bad_orientation.limit_angle=1.1`.

## Key Metrics

Last 100 scalar points:

- `Train/mean_episode_length`: `555.62`
- `Train/mean_reward`: `-0.4727`
- `Curriculum/terrain_levels/mean_terrain_level`: `1.485`
- `Episode_Termination/time_out`: `1.6135`
- `Episode_Termination/base_contact`: `0.001`
- `Episode_Termination/bad_orientation`: `2.183`
- `Episode_Reward/semantic_foot_over_clearance`: `0`
- `Episode_Reward/semantic_body_part_clearance`: `-0.06985`
- `Episode_Reward/base_angular_velocity`: `-0.09671`
- `Episode_Reward/flat_orientation_l2`: `-0.01533`
- `Episode_Reward/feet_slide`: `-0.03666`

Last 20 scalar points:

- `Episode_Termination/bad_orientation`: about `2.43`
- `Episode_Termination/base_contact`: about `0.00375`
- `Episode_Termination/time_out`: about `1.54`
- `Train/mean_episode_length`: about `520.54`
- `mean_terrain_level`: about `1.293`

Saved config snapshot:

- `bad_orientation.limit_angle=1.1`
- `base_angular_velocity.weight=-0.05`
- `flat_orientation_l2.weight=-2.5`
- `feet_slide.weight=-0.1`
- `semantic_foot_over_clearance.weight=1.0`

## Result

The run is not dominated by base contact. `base_contact` is effectively zero near the end, while `bad_orientation` remains consistently nonzero and larger than timeout counts.

The reset threshold was already relatively loose at `1.1 rad` (`~63 deg`). The stronger evidence is that stability rewards were weak compared with the aggressive foot-over reward, and the policy was still at low terrain levels with sparse foot-over reward hits.

## Conclusion

Do not globally relax reset much further as the first fix. A larger global angle threshold would likely let the policy train through near-fall states and can make the learned crossing less stable.

Better first actions:

- keep `bad_orientation` around `1.1`
- strengthen stability shaping
- lower the foot-over bonus weight
- use controlled-crossing reset-stage diagnostics to decide whether falls happen after foot-over or before the obstacle

Only consider a small temporary relaxation such as `1.2` if future reset-stage diagnostics show recoverable post-foot-over resets, not early uncontrolled falls.

## Follow-up

- Use the new controlled-crossing reset diagnostics on the next checkpoint after stability retuning.
- Compare `reset_stage_counts` and `reset_reason_counts` before deciding whether to relax `bad_orientation`.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10/env_cfg.yaml](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10/env_cfg.yaml)
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10/events.out.tfevents.1781669017.enine.1576822.0](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10/events.out.tfevents.1781669017.enine.1576822.0)
