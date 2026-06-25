# Flat-Small 21:46 Checkpoint Eval And Reward Correlation

## Purpose

Evaluate selected checkpoints from run `2026-06-23_21-46-49` and test the user's hypothesis that trying to learn overpass actions caused orientation resets.

## Stage

Checkpoint evaluation / TensorBoard reward correlation / flat-small avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

Controlled crossing eval for checkpoints `14700`, `14800`, `14900`, `16700`, and `17300`:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 1200s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 1000 \
  --run-dir /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-23_21-46-49 \
  --checkpoint model_<ckpt>.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_214649_ckpt_sweep/model_<ckpt>
```

TensorBoard correlation used `EventAccumulator` over the same run's event file.

## Input Conditions

- All eval commands exited `0`.
- `mpc_policy_eval.py` checkpoint lookup required an absolute `--run-dir` because its built-in relative fallback does not search the flat-small experiment directory.

## Key Metrics

| Checkpoint | Opportunity | Root Crossed | Foot Over | Small Contact | Success | Bad Orientation Reset | Timeout | Mean Bad Reset Step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `14700` | 16 | 12 | 1 | 1 | 0 | 11 | 5 | 82.4 |
| `14800` | 14 | 5 | 0 | 2 | 0 | 7 | 9 | 75.9 |
| `14900` | 14 | 8 | 0 | 3 | 0 | 6 | 10 | 46.7 |
| `16700` | 15 | 9 | 0 | 3 | 0 | 10 | 6 | 21.4 |
| `17300` | 16 | 4 | 5 | 15 | 0 | 16 | 0 | 10.5 |

Reward/reset correlation with `bad_orientation` over all scalar steps:

- `Policy/mean_noise_std`: `+0.9205`.
- `Episode_Reward/action_rate`: `+0.8637`.
- `Episode_Reward/feet_slide`: `+0.8530`.
- `Episode_Reward/base_angular_velocity`: `+0.8388`.
- `Episode_Reward/flat_orientation_l2`: `+0.8180`.
- `Episode_Reward/semantic_foot_over_clearance`: `-0.0041`.

## Result

The user's suspicion is partly supported but needs a precise reading: by `model_17300`, the policy produces more foot-over events (`5/16`), but this comes with `15/16` small contacts and `16/16` bad-orientation resets at about step `10.5`. That is not successful overpass learning; it looks like unstable high-risk leg motion / obstacle contact while the base immediately loses orientation.

Earlier checkpoints are also not successful: `14700` has the best root crossing (`12/16`) and low small contact (`1/16`), but still `0` clean successes and `11/16` bad-orientation resets. `14800` and `14900` have fewer bad resets than `14700` but no foot-over events. `16700` is already close to collapse with bad resets around step `21`.

## Conclusion

Do not continue from `17300`. The collapse is not explained by a clean foot-over reward improving at the cost of orientation; instead, effective foot-over reward remains sparse/noisy, while action/noise/stability metrics correlate strongly with bad orientation. The late "foot_over" signal at `17300` is likely a symptom of chaotic leg motion and contact, not the desired behavior.

## Follow-Up

- If selecting a checkpoint for visualization/eval, compare `14700`, `14800`, and `14900`; `16700` is likely too close to collapse.
- If restarting training, control exploration/update aggressiveness before changing reset thresholds: e.g. lower entropy/std growth or learning rate, and evaluate stability before extending.
- Add an eval/diagnostic metric that distinguishes true pre-contact foot clearance from post-contact/near-reset foot-over artifacts.

## Git Refs

- Baseline Ref: `704db79`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../logs/mpc_policy_eval/flat_small_214649_ckpt_sweep](../../logs/mpc_policy_eval/flat_small_214649_ckpt_sweep)
