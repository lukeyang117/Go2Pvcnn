# 2026-07-01 Flat-Small Path Swing-Over Reward

## Purpose

Make the existing `semantic_foot_over_clearance` reward distinguish true MPC-aligned swing-over from contact avoidance or near-footprint clearance.

## Stage

RL reward shaping / flat-small semantic avoidance.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Design and plan:

- [../../docs/superpowers/specs/2026-07-01-flat-small-path-swing-over-reward-design.md](../../docs/superpowers/specs/2026-07-01-flat-small-path-swing-over-reward-design.md)
- [../../docs/superpowers/plans/2026-07-01-flat-small-path-swing-over-reward-plan.md](../../docs/superpowers/plans/2026-07-01-flat-small-path-swing-over-reward-plan.md)

RED command:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_penalizes_crossed_path_small_without_swing_overpass \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_swing_overpass_removes_missed_penalty \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_stance_overpass_does_not_satisfy_swing_overpass \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected failures observed:

- new tensor tests failed because `missed_over_penalty` was not accepted;
- cfg static test failed because dense/near caps still had the old values.

Focused verification:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `176 passed`.

Static verification:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Exit `0`.

```bash
git diff --check -- \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  docs/superpowers/specs/2026-07-01-flat-small-path-swing-over-reward-design.md \
  docs/superpowers/plans/2026-07-01-flat-small-path-swing-over-reward-plan.md
```

Exit `0`.

Real smoke:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 4 \
  --mpc_num_envs 4 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc
```

Result: exit `0`; Reward Manager includes `semantic_foot_over_clearance` at weight `1.0`; one training iteration completed without CUDA/OOM or reward parameter errors.

## Code Change

No new reward term was added.

Existing `semantic_foot_over_clearance` now:

- gates positive reward through MPC reference swing feet when reference contact is available;
- keeps strict over-cell reward as the main success signal;
- samples path-small cells and adds a stricter swing-over score against those footprints;
- keeps dense and near shaping as capped auxiliary signal;
- subtracts `missed_over_penalty` when a crossed path-small sample has no strict swing-over;
- applies `reference_reward_mask` after positive/negative shaping.

Flat-small cfg now uses:

- `semantic_foot_over_clearance.weight=1.0`
- `strict_over_cell_bonus_scale=2.0`
- `dense_approach_bonus_fraction=0.05`
- `strict_near_bonus_fraction=0.25`
- `missed_over_penalty=0.15`
- `root_crossed_margin_m=0.02`

## Result

Implementation and focused verification pass. Behavior improvement is not claimed yet because this still needs a short warm-start run and controlled crossing evaluation.

## Conclusion

This is the intended next training candidate after `2026-06-30_22-43-39`: it should make "root crossed but no foot-over" costly while keeping the same reward term and MPC planner loss surface.

## Follow-Up

Warm-start from the latest stable checkpoint and evaluate controlled crossing early:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 1024 \
  --mpc_num_envs 1024 \
  --max_iterations 5000 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc \
  --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_22-43-39 \
  --checkpoint model_19699.pt \
  --keep_std
```

Evaluate around `model_20000-20200`; stop/retune if `foot_over_count` remains `0` or if `bad_orientation` rises.

## Git Refs

- Baseline Ref: `2c8f1fb`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
