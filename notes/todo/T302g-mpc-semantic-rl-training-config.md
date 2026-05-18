# T302g MPC Semantic RL Training Config

## Current State

- T302g is a child of [T302](T302-mpc-body-leg-height-field-collision-safety.md).
- Goal: promote the T302 MPC collision-safe backend into an independent semantic RL train/play task without changing the existing `teacher_elevation_trajectory` / together defaults.
- Written design: [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
- Implementation plan: [../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md)
- P0 acceptance:
  - independent new config file, train/play experiment, and Gym ids
  - high-resolution `semantic_height_scanner` for MPC and collision reward
  - CNN input `2 x 16 x 16` height + priority semantic map
  - MPC foot-only imitation reward
  - swing/leg collision reward reads current IsaacLab body buffers, not planner FK
  - MPC replan reads current IsaacLab state, not prior MPC cache
  - 4096 real IsaacLab collect-data timing under `10s`
  - T302 strict metrics do not regress

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302g.1 | todo | P0 | Implement independent MPC semantic RL train/play config, helpers, tests, 4096 timing gate, and non-regression evidence | `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`, `Go2Pvcnn/extension/mdp/`, `Go2Pvcnn/tests/` |

## Closed Children Archive

- None yet.

## Related Logs

- [../log/2026-05-18-1036-t302g-mpc-semantic-rl-training-design-and-plan.md](../log/2026-05-18-1036-t302g-mpc-semantic-rl-training-design-and-plan.md)
- T302 strict baseline: [../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md](../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `pending`
- Current Work Ref: `working tree on top of 946811f (2026-05-18 10:36 CST)`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
  - [../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/extension/mdp/observations.py](../../Go2Pvcnn/extension/mdp/observations.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)

## Next Step

- Execute [the implementation plan](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md) task-by-task.

## Node Details

### T302g.1 Independent MPC semantic RL rollout config

- why-created: the user wants MPC integrated into RL training as a new semantic task without disturbing together or T302.
- hypothesis: a high-resolution semantic scanner can serve MPC and collision reward, while a downsampled `2 x 16 x 16` map feeds the CNN; dirty-subset MPC scheduling can keep 4096 collect-data under `10s`.
- evidence: design and plan written; implementation and timing evidence pending.

