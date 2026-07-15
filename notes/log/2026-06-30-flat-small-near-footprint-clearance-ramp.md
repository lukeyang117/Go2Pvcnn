# 2026-06-30 Flat-Small Near-Footprint Clearance Ramp

## Purpose

Fix the latest controlled-crossing failure where `model_17400.pt` from `2026-06-30_12-00-28` is stable and has low small contact, but still has `foot_over=0/16` and negative max clearance.

## Stage

Flat-small semantic avoidance reward shaping / training config.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

TDD RED:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_near_footprint_clearance_ramp_rewards_approach_clearance \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected failures observed:

- `semantic_foot_over_clearance_bonus_from_tensors()` did not accept `strict_near_*` parameters.
- flat-small cfg still did not expose the new near-footprint ramp contract.

GREEN/focused verification:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Result: `24 passed`.

Static verification:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py

git diff --check -- \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py
```

Both exited `0`.

## Input Conditions

Latest behavior evidence:

- [2026-06-30-flat-small-120028-tensorboard-readout-and-eval-blocked.md](2026-06-30-flat-small-120028-tensorboard-readout-and-eval-blocked.md)
- `model_17400.pt`: opportunity `12/16`, root crossed `8/16`, foot_over `0/16`, small contact `1/16`, success `0/12`
- max clearance remained negative.

User requested keeping the larger foot-over reward weight because entropy/exploration is already small.

## Code Change

No new reward term was added. Existing `semantic_foot_over_clearance` now has a third component:

- strict over-cell bonus: unchanged final success target, boosted by `strict_over_cell_bonus_scale`
- near-footprint clearance ramp: rewards swing feet near the sampled small-obstacle footprint when `foot_z - obstacle_top_z` moves from negative toward positive clearance
- dense approach bonus: retained as a smaller auxiliary signal

Flat-small cfg now uses:

- `semantic_foot_over_clearance.weight=1.0`
- `strict_over_cell_bonus_scale=2.0`
- `dense_approach_bonus_fraction=0.10`
- `strict_near_along_margin_m=0.18`
- `strict_near_lateral_margin_m=0.12`
- `strict_near_bonus_fraction=0.60`
- `strict_clearance_ramp_margin_m=0.05`

## Result

Focused tests and static checks pass. The reward should now give a continuous clearance signal before the foot is exactly over the semantic cell, while preserving the final controlled-crossing success criterion.

## Conclusion

Do not continue the previous run blindly. Start a short warm-start continuation with the new reward shape and evaluate early. The first checkpoint decision should be around `model_18000-18200` from the same resume lineage.

## Follow-Up

Suggested continuation command from the existing run/checkpoint:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 1024 \
  --mpc_num_envs 1024 \
  --max_iterations 5000 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc \
  --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-30_12-00-28 \
  --checkpoint model_17600.pt \
  --keep_std
```

Evaluate `model_18000-18200` with controlled crossing on card 2. Stop/retune if `foot_over_count` remains `0` while max clearance is still negative.

## Git Refs

- Baseline Ref: `2c8f1fb`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
