# Semantic Map Contact Collision Design

## Purpose

Write a Chinese design for replacing train-time global semantic filtered contact with ordinary robot contact force plus 0.01m semantic/elevation map inference.

## Stage

Train semantic collision reward and flat-small curriculum collision bookkeeping.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)
- [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)

## Baseline Ref

- Current working tree after flat-small all-column semantic course and goal-anchored command changes.

## Candidate Ref

- Design only; no runtime code changes.

## Key Files

- [../../docs/superpowers/specs/2026-06-13-semantic-map-contact-collision-design.md](../../docs/superpowers/specs/2026-06-13-semantic-map-contact-collision-design.md)
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
- [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)

## Result

Design written. The proposed train path removes `semantic_contact_small/large` from train cfgs and replaces global semantic contact reward/curriculum source with:

```text
contact_forces.data.net_forces_w
+ 0.01m semantic_height_scanner map query
=> inferred small/large semantic collision
```

User clarification folded into the design: train cfgs must not load the global semantic contact sensor configs at scene creation time. It is not enough to disable the reward consumer, because the startup stall is caused by sensor/contact-view initialization.

Second clarification folded in:

- PLAY / VIEWER defaults also must not load `semantic_contact_small/large`.
- The map-contact collision penalty should be combined with the existing `_semantic_body_part_clearance_reward_term()` path at the reward/config level, sharing the same body geometry and 0.01m semantic/elevation map query helpers.

## Verification

Design-only pass; no code executed beyond source reading.

## Follow-Up

After user approval, create an implementation plan and then modify code with tests:

1. tensor helper tests for force+semantic inferred hit;
2. cfg tests proving train cfgs no longer create global semantic contact sensors;
3. 8-env IsaacLab smoke;
4. 1024-env startup verification against the previous simulation-start stall.
