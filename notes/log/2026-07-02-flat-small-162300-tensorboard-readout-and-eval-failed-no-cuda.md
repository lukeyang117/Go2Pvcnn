# 2026-07-02 Flat-Small 16:23 TensorBoard Readout And Eval Failed No CUDA

## Purpose

Inspect the new `2026-07-01_16-23-00` flat-small run and run a controlled-crossing eval.

## Stage

Training metrics / controlled crossing / flat-small semantic avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

TensorBoard was read from:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-07-01_16-23-00
```

Eval command:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 timeout 180s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode controlled_crossing \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir unused \
  --checkpoint /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-07-01_16-23-00/model_19699.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.6 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/flat_small_20260701_162300_model19699_controlled_crossing_card2
```

## Input Conditions

- Candidate run: `2026-07-01_16-23-00`
- Checkpoint tested: `model_19699.pt`
- Interpreter: `env_isaacsim`

## Key Metrics

TensorBoard last sample:

- `Train/mean_episode_length`: `928.13`
- `Curriculum/terrain_levels/mean_terrain_level`: `6.168`
- `Episode_Termination/bad_orientation`: `0.2`
- `Episode_Termination/base_contact`: `0.0`
- `Policy/mean_noise_std`: `0.3387`
- `Episode_Reward/semantic_foot_over_clearance`: `0.0` last sample, `670/5000` nonzero overall
- `Episode_Reward/semantic_body_part_clearance`: `-0.2852`
- `Episode_Reward/reference_foot_pos`: `0.0434`
- `Episode_Reward/reference_contact`: `0.00383`
- `Episode_Reward/flat_orientation_l2`: `-0.0123`
- `Episode_Reward/base_angular_velocity`: `-0.1308`
- `Episode_Reward/feet_slide`: `-0.0683`

Eval result:

- IsaacLab started, but aborted before rollout.
- `mpc_policy_eval.py` ended with `RuntimeError('No CUDA GPUs are available')`.
- The session could not verify controlled crossing behavior from this environment.

## Result

Diagnostic only. The run itself looks more stable than the collapsed runs, but the strict eval could not execute in this Codex session because IsaacLab saw no CUDA GPU.

## Conclusion

The new checkpoint is worth testing on a machine/session where IsaacLab can actually enumerate CUDA, but this attempt does not produce a behavioral verdict.

## Follow-Up

- Re-run the same controlled-crossing eval in a GPU-visible shell.
- If the GPU shell is fixed, compare `model_18200` and `model_19699` against the previous `2026-06-30_22-43-39` run.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-07-01_16-23-00/train_cfg.yaml](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-07-01_16-23-00/train_cfg.yaml)
  - [../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-07-01_16-23-00/env_cfg.yaml](../../logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-07-01_16-23-00/env_cfg.yaml)
