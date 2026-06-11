# T302r Go2 Geometry Clearance Implementation

## Purpose

Implement the Go2 body-geometry `semantic_body_part_clearance_reward` path and verify it locally plus in a small real IsaacLab smoke.

## Stage

RL reward implementation / flat-small semantic clearance.

## Related Todo

- [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)
- Parent: [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Implemented:

- foot sphere neighborhood queries
- calf/thigh capsule section neighborhood queries
- base oriented footprint neighborhood queries
- cached circular offsets per radius/resolution/device/dtype
- geometry aggregation over current scanner semantic/elevation maps
- flat-small cfg geometry params

Verification commands:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py -q
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py -q
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance --num_envs 16 --max_iterations 1 --headless
```

An initial real smoke using `--task Isaac-Teacher-ManagerBased-RslRl-Go2-FlatSmallAvoidance-v0` failed at CLI parsing because current `train.py` uses `--experiment`; the corrected `--experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance` command passed.

## Input Conditions

- Current reward was point-only and could stay at zero despite sparse true semantic contact.
- Flat-mask/curriculum bookkeeping was already fixed before this implementation.
- User requested code implementation after the standalone T302r plan.

## Key Metrics

- RED: focused test collection failed on missing `_body_geometry_query_points`, proving new geometry tests targeted missing behavior.
- Reward tests after implementation: `14 passed`.
- Reward + backend static tests: `158 passed, 1 warning`.
- Reward + semantic curriculum focused tests: `34 passed`.
- `py_compile`: exit `0`.
- Real IsaacLab smoke: exit `0`.
- Real smoke log dir: `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-11_15-50-40`.
- Real smoke active reward terms: `19`, including `semantic_body_part_clearance`.
- Real smoke env cfg includes `calf_sections: 7`, `thigh_sections: 7`, `base_footprint_grid: (5, 3)`, and `include_base: true`.

## Result

Pass for local implementation and 16-env real smoke. The reward path now uses geometry neighborhoods instead of sparse point-only hits.

## Conclusion

The implementation preserves the current scanner-map contract and keeps the hot path batched in tensors. It does not modify MPC planner loss/reference/command shaping, and it keeps `semantic_contact_collision` as the real-contact signal.

## Follow-Up

- Run a larger 64/1024-env real probe to measure nonzero `semantic_body_part_clearance` rate and collection time.
- Run a short resume training/TensorBoard sanity check before claiming behavior improvement.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree on 2026-06-11 15:51
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
