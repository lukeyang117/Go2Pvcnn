# 2026-06-26 Flat-Small 21:47 TensorBoard And Checkpoint Eval

## Purpose

Evaluate flat-small avoidance run `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-24_21-47-13` after restoring finite `bad_orientation` reset and lowering flat-small `entropy_coef` to `0.002`.

## Stage

Training metrics, controlled crossing checkpoint evaluation, and short MPC tracking overlap.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Commands

TensorBoard scalar readout used `EventAccumulator` over the run directory.

Controlled crossing outputs inspected:

- `logs/mpc_policy_eval/flat_small_20260624_214713_model19300_controlled_crossing_absckpt/2026-06-26_19-08-06-598774/summary.json`
- `logs/mpc_policy_eval/flat_small_20260624_214713_model19699_controlled_crossing_absckpt/2026-06-26_19-08-06-637271/summary.json`

Short tracking command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-24_21-47-13/model_19699.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260624_214713_model19699_tracking_20step
```

## Input Conditions

- Saved train tag records `mpc num_env 2048`, `bad_orientation.limit_angle=1.1`, `semantic_foot_over_clearance=0.12`, strengthened orientation/angular/feet-slide penalties, and `entropy_coef=0.002`.
- Run scalar step range is `14700..19699`.
- Checkpoints evaluated: `model_19300.pt` and `model_19699.pt`.

## Key Metrics

TensorBoard last-100 metrics:

- `Train/mean_episode_length`: `951.038`
- `Train/mean_reward`: `6.75178`
- `Curriculum/terrain_levels/mean_terrain_level`: `6.09475`
- `Policy/mean_noise_std`: `0.318864`
- `Episode_Termination/bad_orientation`: `0.16375`
- `Episode_Termination/base_contact`: `0.00525`
- `Episode_Reward/reference_foot_pos`: `0.04718`
- `Episode_Reward/reference_contact`: `0.00413`
- `Episode_Reward/semantic_body_part_clearance`: `-0.22190`
- `Episode_Reward/semantic_foot_over_clearance`: last-100 `0.0`, nonzero `19/5000`, max `0.01695`

Controlled crossing:

- `model_19300.pt`: opportunity `13/16`, root crossed `6/16`, foot-over `0/16`, small contact `0/16`, overpass success `0/13`, reset `0/16`.
- `model_19699.pt`: opportunity `16/16`, root crossed `8/16`, foot-over `0/16`, small contact `1/16`, overpass success `0/16`, bad-orientation reset `3/16`.

Tracking smoke for `model_19699.pt`:

- mean foot tracking error `0.09437m`
- p95 `0.20033m`
- reference valid ratio `1.0`

## Result

Diagnostic pass. The lower entropy run is much more stable than the two prior collapsing runs, but it still does not learn clean low-small foot-over behavior.

## Conclusion

Do not treat `model_19699.pt` as a solved crossing policy. The main improvement is stability: std stays near `0.32`, episode length remains near `950`, terrain level remains around `6`, and base-contact resets stay low. The remaining failure is signal/behavioral: `semantic_foot_over_clearance` is almost absent in TensorBoard and controlled crossing has `0` foot-over events for both tested checkpoints.

The next code change should focus on making the foot-over teaching signal denser or better aligned with approach/step timing, while preserving `entropy_coef=0.002` and finite `bad_orientation`.

## Follow-Up

- Inspect whether the foot-over reward trigger is too strict relative to the learned gait and map-contact clearance values.
- Consider a pre-foot-over shaping term or staged obstacle/opportunity curriculum before increasing reward weight again.
- Use `model_19300.pt` if a stable visual checkpoint is needed; use `model_19699.pt` only as a latest-stability candidate, not as an overpass success candidate.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/agent/train_cfg.py](../../Go2Pvcnn/agent/train_cfg.py)
