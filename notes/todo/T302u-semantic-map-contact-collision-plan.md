# T302u Semantic Map Contact Collision Plan

## Current State

Global semantic filtered contact became too expensive after flat-small objects were correctly generated across all 20 columns. The train/play/viewer default path must stop loading `semantic_contact_small` and `semantic_contact_large`.

## Goal

Use ordinary robot `contact_forces` plus the 0.01m semantic/elevation scanner map to infer semantic collisions, and fold the penalty into the existing body-part clearance reward.

## Open Children

- [x] T302u.1 Add tensor-level map-contact inference tests and helper.
- [x] T302u.2 Combine map-contact penalty into `semantic_body_part_clearance_reward`.
- [x] T302u.3 Replace curriculum small-collision bookkeeping source with map-contact inference.
- [x] T302u.4 Remove global semantic contact sensor loading from train/play/viewer defaults.
- [x] T302u.5 Run focused local tests, pycompile, and an `env_isaacsim` smoke.
- [ ] T302u.6 Run a 1024-env startup check after older stuck jobs are stopped.

## Key Files

- [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
- [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
- [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
- [../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py](../../Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py)
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

## Related Design

- [../../docs/superpowers/specs/2026-06-13-semantic-map-contact-collision-design.md](../../docs/superpowers/specs/2026-06-13-semantic-map-contact-collision-design.md)

## Related Logs

- [../log/2026-06-13-2348-semantic-map-contact-collision-design.md](../log/2026-06-13-2348-semantic-map-contact-collision-design.md)
- [../log/2026-06-14-0008-flat-small-1024-simulation-start-stall.md](../log/2026-06-14-0008-flat-small-1024-simulation-start-stall.md)
- [../log/2026-06-14-0035-semantic-map-contact-collision-implementation.md](../log/2026-06-14-0035-semantic-map-contact-collision-implementation.md)

## Next Step

Run a 1024-env startup check when GPU/process state is clean enough for a fair measurement.
