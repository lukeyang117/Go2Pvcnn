# 2026-06-27 Flat-Small 19:34 Checkpoint Eval

## Purpose

Evaluate several checkpoints from dense foot-over run `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-26_19-34-12` to see whether the denser TensorBoard `semantic_foot_over_clearance` signal produces actual controlled low-small crossing behavior.

## Stage

Checkpoint evaluation / controlled crossing / dense foot-over reward diagnosis.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Ran `Go2Pvcnn/scripts/mpc_policy_eval.py --mode controlled_crossing` under `env_isaacsim` with:

- `--num-envs 16`
- `--num-rounds 1`
- `--max-steps 300`
- fixed command `0.6 0.0 0.0`
- flat-small controlled crossing terrain selection through `--terrain-rows 0 --terrain-cols 0`

Evaluated checkpoints:

- `model_18600.pt`
- `model_19300.pt`
- `model_19699.pt`

## Input Conditions

- Candidate run: `2026-06-26_19-34-12`
- This run has dense approach `semantic_foot_over_clearance` active.
- TensorBoard already showed dense foot-over scalar but unstable high-entropy training:
  - `semantic_foot_over_clearance` nonzero `3279/5000`
  - `entropy_coef=0.01`
  - repeated bad-orientation collapse buckets

## Commands

Representative command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 600s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 300 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-26_19-34-12/model_19300.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.6 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260626_193412_model19300_controlled_crossing_absckpt
```

## Output Directories

- `logs/mpc_policy_eval/flat_small_20260626_193412_model18600_controlled_crossing_absckpt/2026-06-27_20-39-06-856808`
- `logs/mpc_policy_eval/flat_small_20260626_193412_model19300_controlled_crossing_absckpt/2026-06-27_20-42-08-354125`
- `logs/mpc_policy_eval/flat_small_20260626_193412_model19699_controlled_crossing_absckpt/2026-06-27_20-39-06-719870`

## Key Metrics

| Checkpoint | Opportunity | Root crossed | Foot-over | Small contact | Touchdown on small | Success | Resets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `18600` | `13/16` | `6/16` | `0/16` | `0/16` | `0/16` | `0/13` | `bad_orientation=3`, `base_contact=1` |
| `19300` | `14/16` | `5/16` | `0/16` | `2/16` | `2/16` | `0/14` | `bad_orientation=8` |
| `19699` | `15/16` | `7/16` | `0/16` | `3/16` | `1/16` | `0/15` | `bad_orientation=12` |

Baseline comparison:

- Stable lower-entropy run `2026-06-24_21-47-13/model_19300.pt`: `foot_over=0/16`, small contact `0/16`, success `0/13`, reset `0/16`.
- Stable lower-entropy run `2026-06-24_21-47-13/model_19699.pt`: `foot_over=0/16`, small contact `1/16`, success `0/16`, bad-orientation reset `3/16`.
- Older `2026-06-17_12-01-10/model_14700.pt`: `foot_over=2/16`, small contact `3/16`, success `0/16`.

## Result

Diagnostic pass. These candidate checkpoints do not show behavior improvement in controlled crossing.

The dense reward clearly made the scalar denser during training, but in this high-entropy run the evaluated policies still have `foot_over=0/16`. Later checkpoints add small-object contact and bad-orientation resets rather than clean overpass.

## Conclusion

Do not continue the `2026-06-26_19-34-12` run from `model_19699.pt`. Among this run's checkpoints, `model_18600.pt` remains the least bad diagnostic candidate because it has fewer contacts/resets, but it also has no measured foot-over behavior.

The useful signal from this run is not the policy checkpoint; it is the reward-shaping evidence. The next proper training attempt should use dense foot-over reward with the lower flat-small `entropy_coef=0.002` setting, then re-run controlled crossing early and require nonzero `foot_over_count` without a rise in bad-orientation resets.

## Follow-Up

- Start a fresh/continued run with dense reward plus low entropy.
- Stop early if `semantic_foot_over_clearance` becomes dense only by increasing `bad_orientation` or small contact.
- Use controlled crossing `foot_over_count`, `small_contact_env_count`, and `reset_reason_counts` as the behavior gate, not TensorBoard scalar alone.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
