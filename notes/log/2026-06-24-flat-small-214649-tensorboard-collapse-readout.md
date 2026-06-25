# Flat-Small 21:46 TensorBoard Collapse Readout

## Purpose

Analyze whether run `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-23_21-46-49` should continue training.

## Stage

Training metrics / flat-small avoidance / resume from `model_14700.pt`.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

Used TensorBoard `EventAccumulator` to read scalar tags from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-23_21-46-49/events.out.tfevents.1782222556.enine.1884756.0
```

Checked scalar summaries and 100-iteration buckets for episode length, reward, terrain level, reset reasons, semantic rewards, reference rewards, std, and value loss.

## Input Conditions

- Run directory contains checkpoints `model_14700.pt` through `model_17300.pt`.
- Saved config has `scene.num_envs=2048` and `mpc_planner_cfg.runtime.parallel_plan_batch_size=2048`.
- `tag.txt` says speed curriculum removed, `bad_orientation.limit_angle=1.1`, `semantic_foot_over_clearance` weight lowered `1.0 -> 0.12`, `flat_orientation_l2=-3.5`, `base_angular_velocity=-0.12`, `feet_slide=-0.18`, and `action_rate` unchanged.

## Key Metrics

- Last scalar step: `17302`.
- `Train/mean_episode_length`: last100 `12.55`, last `12.66`.
- `Episode_Termination/bad_orientation`: last100 `161.48`, last `159.63`.
- `Curriculum/terrain_levels/mean_terrain_level`: last100 `0`, last `0`.
- `Episode_Termination/base_contact`: last100 `0.01775`, near zero.
- `Episode_Reward/semantic_foot_over_clearance`: nonzero `113/2603`, last100 `6.8e-06`.
- `Episode_Reward/reference_foot_pos`: last100 `9.06e-05`, effectively gone after collapse.
- `Policy/mean_noise_std`: last100 `0.889`, last `0.893`.
- `Loss/value_function`: last100 `55.35`, elevated after collapse.
- Bucket trend: `14700-14899` reached episode length `558-617` and terrain `0.87-1.75`; by `16800-16899`, episode length dropped to `108` and bad orientation rose to `49.9`; from `16900` onward terrain stayed `0` and episode length stayed `12-17`.

## Result

This run has collapsed into short bad-orientation resets. It is not merely failing to learn low-small overpass; the locomotion policy became unstable and curriculum fell fully back to row `0`.

## Conclusion

Do not continue this exact run from `model_17300.pt`. The better checkpoint window is before collapse, likely around `model_14800.pt` to `model_14900.pt` for early high terrain, or around `model_16700.pt` as the last pre-collapse region, but behavior eval is needed before choosing.

## Follow-Up

Run controlled crossing / tracking eval on pre-collapse checkpoints (`14800`, `14900`, `16700`) before any long continuation. If restarting training, lower exploration pressure or use a smaller/safer resume setup rather than continuing from the collapsed checkpoint.

## Git Refs

- Baseline Ref: `704db79`
- Candidate Ref: working tree
- Key Files:
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-23_21-46-49](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-23_21-46-49)
