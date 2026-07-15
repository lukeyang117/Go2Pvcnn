# 2026-07-01 Flat-Small 22:43 TensorBoard And Card1 Eval

## Purpose

Evaluate the new `2026-06-30_22-43-39` warm-start run after the near-footprint clearance ramp change, and verify whether the current Codex session can see GPU 1 for IsaacLab evaluation.

## Stage

Training metrics / controlled crossing / flat-small semantic avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

TensorBoard was read with `EventAccumulator` from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_22-43-39
```

GPU visibility was checked with:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Controlled crossing was run on card 1:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 timeout 900s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 16 \
  --num-rounds 1 \
  --max-steps 300 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_22-43-39/model_18200.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.6 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260630_224339_model18200_controlled_crossing_card1
```

The same command was then run with `model_19699.pt` and output dir:

```text
logs/mpc_policy_eval/flat_small_20260630_224339_model19699_controlled_crossing_card1
```

## Input Conditions

- Candidate run: `2026-06-30_22-43-39`
- Reward Manager in eval has `semantic_foot_over_clearance` weight `1.0`
- Previous strict run `2026-06-30_12-00-28/model_17400.pt` had `foot_over=0/16`, small contact `1/16`, and success `0/12`

## Key Metrics

TensorBoard last100:

- `Train/mean_episode_length`: `958.8007`
- `Curriculum/terrain_levels/mean_terrain_level`: `6.0671`
- `Episode_Termination/bad_orientation`: `0.1505`
- `Episode_Termination/base_contact`: `0.00125`
- `Policy/mean_noise_std`: `0.35594`
- `Episode_Reward/semantic_foot_over_clearance`: `0.00908`, nonzero `373/5000`, max `1.12665`
- `Episode_Reward/semantic_body_part_clearance`: `-0.29641`
- `Episode_Reward/reference_foot_pos`: `0.04804`
- `Episode_Reward/reference_contact`: `0.00418`

Controlled crossing:

| Checkpoint | Opportunity | Root Crossed | Foot Over | Small Contact | Success | Reset |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `model_18200.pt` | `12/16` | `4/16` | `0/16` | `3/16` | `0/12` | `bad_orientation=1` |
| `model_19699.pt` | `12/16` | `6/16` | `0/16` | `0/16` | `0/12` | `bad_orientation=1` |

GPU status:

- GPU 1 was visible and essentially idle before eval.
- IsaacLab eval successfully created the environment on `cuda:0` under `CUDA_VISIBLE_DEVICES=1`.
- Omniverse still printed CUDA enumeration warnings, but the run completed and produced summaries.

## Result

Diagnostic. The run is stable and no longer shows the previous collapse pattern. The near-footprint ramp makes the TensorBoard foot-over scalar denser than the previous strict run, and the latest checkpoint reduces small-object contact to `0/16` in this controlled crossing sample.

However, true controlled-crossing foot-over remains absent: both tested checkpoints have `foot_over=0/16` and success `0/12`.

## Conclusion

Do not continue this run blindly expecting the current reward to become a reliable stepping-over policy. The latest checkpoint is better at avoiding contact, but it still has not learned to lift feet over low-small cells. The next code direction should separate "avoid touching small objects" from "must execute MPC-aligned swing-over when root path crosses a small object."

## Follow-Up

- Add or tighten diagnostics for root-crossed-but-no-foot-over cases.
- Consider making the existing reward/curriculum require foot-over success in path-obstacle opportunities instead of rewarding near-footprint clearance that can be satisfied while sidestepping or avoiding.
- Keep finite `bad_orientation` reset; this run is not failing primarily by orientation collapse.

## Git Refs

- Baseline Ref: `2c8f1fb`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
