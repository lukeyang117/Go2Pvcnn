# 2026-06-26 Flat-Small Dense Foot-Over Reward

## Purpose

Make the existing `semantic_foot_over_clearance` reward less sparse for flat-small avoidance. The previous implementation only rewarded a foot after it was already directly above a low-small semantic cell, which matched the observed TensorBoard sparsity in run `2026-06-24_21-47-13` (`19/5000` nonzero, last100 `0`).

## Stage

RL reward shaping for flat-small semantic avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Code Changes

- [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - `semantic_foot_over_clearance_bonus_from_tensors()` now samples the command-heading corridor ahead of the robot, detects low-small semantic cells, and gives a continuous approach clearance bonus when a swing foot is close to the obstacle and high enough.
  - Existing over-cell bonus is preserved, but when MPC reference contact is available it is also gated to reference swing feet.
  - Optional `reference_contact_state` gates dense and over-cell foot-over reward to MPC swing legs.
  - Optional `reference_reward_mask` suppresses reward for envs not participating in MPC reference reward.
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - Flat-small `semantic_foot_over_clearance` params now explicitly include approach sampling and Gaussian proximity widths.
- Tests:
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

## Input Conditions

- Existing reward term name and weight are unchanged: `semantic_foot_over_clearance`, weight `0.12`.
- No MPC planner loss or loss key was added.
- The new dense reward uses current scanner terrain, current foot positions, command heading, and MPC reference contact state when available.

## Verification

RED:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Observed before implementation: `2 failed, 19 passed`; failures were missing `reference_contact_state` and dense approach parameters.

GREEN:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
```

Observed: `21 passed`.

Config RED/GREEN:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Observed before config wiring: `KeyError: 'approach_sample_count'`. Observed after wiring: `1 passed`.

Focused verification:

```bash
pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_train_cfg_uses_lower_entropy_without_affecting_base \
  -q
```

Observed: `23 passed`.

Pycompile and whitespace:

```bash
python -m py_compile \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
git diff --check
```

Observed: both exit `0`.

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 240s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --device cuda:0 \
  --num_envs 4 \
  --mpc_num_envs 4 \
  --max_iterations 1 \
  --planner-backend mpc
```

Observed exit `0`. Reward Manager includes `semantic_foot_over_clearance` weight `0.12`; Curriculum Manager has only `terrain_levels`; one PPO iteration completes.

## Result

Pass. The reward is now dense enough in principle to fire before the foot is exactly over the obstacle cell, and it is gated by MPC reference swing/contact state when available.

## Conclusion

The implementation addresses the immediate sparsity root cause without adding a new reward term or changing MPC planner losses. The next required evidence is a short warm-start training run to verify TensorBoard `Episode_Reward/semantic_foot_over_clearance` becomes meaningfully nonzero while `bad_orientation` and `base_contact` stay controlled.

## Follow-Up

- Resume from a stable checkpoint such as `2026-06-24_21-47-13/model_19300.pt` or `model_19699.pt` and run a short training segment.
- Watch `semantic_foot_over_clearance` nonzero frequency, `bad_orientation`, `base_contact`, `mean_episode_length`, and terrain level.
- Re-run controlled crossing and require `foot_over_count > 0` before judging success.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
