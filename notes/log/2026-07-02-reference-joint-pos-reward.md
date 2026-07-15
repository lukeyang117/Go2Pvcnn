# 2026-07-02 Reference Joint Pos Reward

## Purpose

Add MPC joint-angle tracking information to the RL reward path so the policy can learn the leg pose implied by the MPC trajectory, not only foot position/contact.

## Stage

MPC reference reward / flat-small semantic avoidance training.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

RED:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_rl_participation.py::test_reference_joint_pos_reward_uses_current_mpc_frame_and_reward_mask \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_teacher_mpc_semantic_cfg_enables_small_weight_reference_joint_pos_reward \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner -q
```

Expected failures observed:

- `reference_joint_pos_reward` did not apply `reference_reward_mask`.
- semantic MPC cfg did not wire `reference_joint_pos`.
- PLAY cfg did not explicitly disable it.

GREEN/focused:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_rl_participation.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `156 passed, 1 skipped`.

Static:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Exit `0`.

```bash
git diff --check -- \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/tests/test_mpc_rl_participation.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
```

Exit `0`.

## Code Change

- Existing `reference_joint_pos_reward` now applies `reference_reward_mask`, matching foot/contact reference rewards.
- `TeacherElevationTrajectoryMpcSemanticRewardsCfg` now includes `reference_joint_pos` with weight `0.05`.
- MPC-enabled train/viewer configs keep the reward enabled.
- no-MPC PLAY configs explicitly set `reference_joint_pos=None`.

## Result

Implementation and focused verification pass. Real IsaacLab smoke was not run in this Codex session because prior attempts in this session could not enumerate CUDA.

## Conclusion

The training signal now includes MPC joint-angle tracking. This is intended to give the policy the leg-pose information needed for swing-over, instead of only endpoint/contact targets.

## Follow-Up

- Run a short warm-start from `2026-07-01_16-23-00/model_19699.pt` with this reward.
- Watch `Episode_Reward/reference_joint_pos`, `bad_orientation`, and controlled-crossing `foot_over_count`.
- If joint tracking makes gait stiff, reduce weight from `0.05` to `0.02`.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_mpc_rl_participation.py](../../Go2Pvcnn/tests/test_mpc_rl_participation.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
