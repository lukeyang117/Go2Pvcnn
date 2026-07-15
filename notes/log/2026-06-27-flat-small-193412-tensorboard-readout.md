# 2026-06-27 Flat-Small 19:34 TensorBoard Readout

## Purpose

Inspect new flat-small avoidance run `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-26_19-34-12` after dense approach `semantic_foot_over_clearance` reward was added.

## Stage

Training metrics / TensorBoard scalar diagnosis.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Used TensorBoard `EventAccumulator` to read all scalar points from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-26_19-34-12/events.out.tfevents.1782473811.enine.3205177.0
```

Also checked:

- `tag.txt`
- `train_cfg.yaml`
- `env_cfg.yaml` snippets with `rg`

## Input Conditions

- Scalar range: steps `14700..19699`, `5000` points.
- `tag.txt` says dense foot-over approach reward is enabled.
- `env_cfg.yaml` confirms:
  - `semantic_foot_over_clearance` has `approach_sample_count: 9`
  - `approach_min_distance_m: 0.05`
  - `approach_along_sigma_m: 0.18`
  - `approach_lateral_sigma_m: 0.16`
  - `bad_orientation.limit_angle: 1.1`
- `train_cfg.yaml` confirms:
  - `algorithm.entropy_coef: 0.01`
  - `policy.init_noise_std: 1.0`

## Key Metrics

Last-100:

- `Train/mean_episode_length`: `545.70`
- `Train/mean_reward`: `-4.824`
- `Curriculum/terrain_levels/mean_terrain_level`: `1.384`
- `Episode_Termination/bad_orientation`: `2.126`
- `Episode_Termination/base_contact`: `0.0200`
- `Policy/mean_noise_std`: `0.5667`
- `Episode_Reward/semantic_foot_over_clearance`: `0.003781`
- `Episode_Reward/semantic_body_part_clearance`: `-0.1094`
- `Episode_Reward/reference_foot_pos`: `0.1016`
- `Episode_Reward/reference_contact`: `0.00888`

Whole-run foot-over signal:

- `semantic_foot_over_clearance` nonzero `3279/5000`
- previous stable lower-entropy run `2026-06-24_21-47-13` had only `19/5000` nonzero and last100 `0`

Bucket pattern:

- `15000-15999`: severe collapse, episode length `61.34 -> 16.27`, bad_orientation `130-133`, terrain falls to `0`.
- `18000-18999`: partial recovery, episode length about `585-599`, terrain about `1.66-1.69`, bad_orientation about `1.86-1.98`.
- `19000-19699`: foot-over scalar becomes larger (`0.0036-0.0038`) but reward is negative and terrain remains low.

Best 100-step windows:

- Stable-best around `model_18600`: `episode_length=670.77`, terrain `1.895`, bad_orientation `1.354`, foot-over `0.001617`, reward `2.139`, std `0.483`.
- Highest useful foot-over windows with `episode_length > 400` are around `model_19100-19400`, but they have bad_orientation around `2.15-2.38`, base_contact around `0.03-0.038`, and negative reward.

Correlations with bad_orientation:

- `Policy/mean_noise_std`: `+0.830`
- `Train/mean_episode_length`: `-0.863`
- `semantic_foot_over_clearance`: `-0.367`
- `semantic_body_part_clearance`: `+0.229`

## Result

Diagnostic pass. Dense foot-over reward did what it was supposed to do at the signal level: the scalar is now dense and trends upward. However, this run is not a stable policy-improvement run because `entropy_coef` returned to `0.01`, noise/std is higher than the stable `0.002` run, and the run repeatedly collapses into bad-orientation resets.

## Conclusion

Do not continue blindly from `model_19699.pt`. If selecting a checkpoint from this run, `model_18600.pt` is the most defensible diagnostic candidate because it has the best short-window stability while already showing denser foot-over signal.

The next proper training attempt should combine:

- dense foot-over reward from this run
- flat-small `entropy_coef=0.002` from the stable `2026-06-24_21-47-13` run
- finite `bad_orientation.limit_angle=1.1`

Then verify whether `semantic_foot_over_clearance` stays dense without the mid-run bad-orientation collapse.

## Follow-Up

- Evaluate `model_18600.pt` and latest `model_19699.pt` in controlled crossing only if behavior evidence is needed before another training run.
- Prefer retraining/resuming with `entropy_coef=0.002` before long continuation.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/agent/train_cfg.py](../../Go2Pvcnn/agent/train_cfg.py)
