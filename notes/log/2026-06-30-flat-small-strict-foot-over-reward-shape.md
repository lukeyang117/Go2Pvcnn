# 2026-06-30 Flat-Small Strict Foot-Over Reward Shape

## Purpose

Change the existing `semantic_foot_over_clearance` reward so the true over-cell clearance path is distinct from, and stronger than, the dense approach shaping path.

## Stage

Flat-small semantic avoidance reward shaping / training config.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Procedure

Implemented a TDD change after the `2026-06-29_15-34-52` run showed that raising `semantic_foot_over_clearance.weight` to `1.0` increased the TensorBoard scalar without producing controlled-crossing `foot_over`.

RED:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py::test_foot_over_bonus_separates_strict_over_cell_from_capped_dense_approach \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Expected failures observed:

- `strict_over_cell_bonus_scale` did not exist.
- cfg weight was still `1`, while the new contract expects `0.6`.

GREEN/focused verification:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract -q
```

Result: `23 passed`.

Static checks:

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

Previous evidence:

- [2026-06-30-flat-small-153452-foot-over-weight1-eval.md](2026-06-30-flat-small-153452-foot-over-weight1-eval.md) showed `weight=1.0` made the scalar strong but controlled crossing still had `foot_over=0/16` and success `0`.

## Code Change

- `semantic_foot_over_clearance_bonus_from_tensors` now accepts:
  - `strict_over_cell_bonus_scale`
  - `dense_approach_bonus_fraction`
- The strict over-cell branch is multiplied by `strict_over_cell_bonus_scale`.
- The dense approach branch is capped to `dense_approach_bonus_fraction * bonus_clip`, so it remains a shaping signal rather than a substitute for true over-cell clearance.
- The flat-small config now uses:
  - `semantic_foot_over_clearance.weight=0.6`
  - `strict_over_cell_bonus_scale=2.0`
  - `dense_approach_bonus_fraction=0.20`

## Result

Focused tests and static checks pass. No new reward term or MPC planner loss was added.

## Conclusion

The training reward now separates "early approach lift" from "true over-cell clearance". This should reduce the failure mode where the policy collects dense foot-over scalar without satisfying the controlled-crossing `foot_over` metric.

## Follow-Up

- Run a short warm-start training from the current flat-small checkpoint and check:
  - `semantic_foot_over_clearance` remains nonzero;
  - `bad_orientation` and `base_contact` do not rise sharply;
  - controlled crossing shows positive `foot_over_count` and improved max clearance.
- This log does not claim behavior improvement yet; no IsaacLab training/eval smoke was run for the new reward shape.

## Git Refs

- Baseline Ref: `2c8f1fb`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
