# T302r Go2 Geometry Clearance Reward Plan

## Purpose

Create a standalone todo/plan file for implementing the Go2 body-geometry version of `semantic_body_part_clearance_reward`.

## Stage

RL reward implementation planning / flat-small semantic clearance.

## Related Todo

- [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)
- Parent: [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Used the approved Chinese HTML design:

- [../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html](../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html)

Read current reward and test boundaries:

- [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
- [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)

Created:

- [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)

## Input Conditions

- User asked: "写新的todo文件md，当作plan".
- The design was already written in Chinese HTML and logged.
- No runtime code implementation was requested in this turn.

## Key Metrics

- New plan tasks: `7`.
- Open children: `5`.
- Main runtime files mapped: `3`.
- Required real verification stages: local pytest, fixed-shape/perf guard, real `env_isaacsim` probe, short TensorBoard sanity.

## Result

Pass. A standalone plan now exists for the Go2 geometry clearance reward implementation.

## Conclusion

T302r separates the next reward implementation from the broader T302q flat-small training/curriculum plan. It keeps the same hard constraints: no MPC planner changes, no per-env Python loop, no reward-side scanner anchor cache, and no USD/PhysX geometry queries in the hot path.

## Follow-Up

- Start T302r Task 1 with RED tests before modifying runtime reward code.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree on 2026-06-11 15:19
- Key Files:
  - [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)
  - [../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html](../../docs/superpowers/specs/2026-06-11-go2-body-geometry-clearance-reward-design.html)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
