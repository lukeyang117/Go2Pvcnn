# Flat-small foot-over training signal

## Purpose

根据 `model_23600.pt` controlled crossing eval 的结果，直接修改 flat-small 训练配置，使低难度阶段更容易产生路径小障碍机会，并新增足端跨越小障碍的正向奖励信号。

## Stage

- RL config / flat-small semantic avoidance reward shaping
- Related todo: [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Code Changes

- [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - Added `semantic_foot_over_clearance_bonus_from_tensors()`.
  - Added `semantic_foot_over_clearance_bonus()` env reward wrapper.
  - The reward is positive only when a foot is over a low-small semantic cell in the commanded path corridor and clears the cell height by the configured margin.
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - Flat-small config now mounts `semantic_foot_over_clearance`.
  - Low rows use smaller center safety holes and lower spacing to increase path-obstacle teaching opportunities.
  - Small contact penalty is strengthened by `small_weight=2.5` and `force_scale=25.0`.

## Verification

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Result: `18 passed`.

```bash
python -m py_compile Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Result: exit `0`.

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

Result: exit `0`; Reward Manager has `20` terms including `semantic_foot_over_clearance`; Curriculum Manager still only has `terrain_levels`.

## Key Metrics

- Smoke output directory: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-12_18-33-38`
- `Episode_Reward/semantic_foot_over_clearance`: `0.0000` in the one-iteration smoke.
- `Curriculum/terrain_levels/mean_terrain_level`: `0.4563`
- Collection: `5.735s`

## Conclusion

The flat-small training config now has an explicit foot-over teaching reward and denser low-level path opportunities. The one-iteration smoke proves the term wires into IsaacLab and does not break training startup, but it is too short to prove the reward becomes nonzero during learning.

## Follow-up

- Start a short warm-start run and watch `Episode_Reward/semantic_foot_over_clearance`.
- Re-run controlled crossing eval after a small amount of training; expected first sign of progress is `foot_over_count > 0`, even before overpass success becomes high.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
