# T302g MPC Semantic RL Training Design And Plan

## Purpose

Record the approved design and implementation plan for a new independent MPC semantic RL train/play task.

## Stage

Design and planning for RL task configuration, semantic scanner observation, swing/leg collision reward, MPC dirty-subset runtime, and 4096 collect-data timing acceptance.

## Related Todo

- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
- Parent: [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Baseline Ref

- Working tree on top of `946811f`

## Candidate Ref

- Documentation-only working tree on top of `946811f`

## Key Files

- [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
- [../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md)
- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
- [../todo.md](../todo.md)
- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)
- [index.md](index.md)

## Requirements Captured

- New task config must live in a new Python file.
- Existing `teacher_elevation_trajectory` / together path must not be changed.
- The new task defaults to MPC.
- Add high-resolution `semantic_height_scanner`.
- CNN input remains the same spatial size as current trajectory config: `2 x 16 x 16`.
- Semantic downsampling keeps priority: large obstacle > small obstacle > terrain.
- MPC imitation reward is foot-only.
- `swing_leg_collision_reward` reads current IsaacLab body/link buffers and does not recompute FK.
- MPC replans read current IsaacLab robot state, not the previous MPC cache.
- Reset/velocity-change replans must select dirty env subsets, not replan all 4096 envs together.
- A real 4096 IsaacLab collect-data timing test must keep each measured pass under `10s`.
- T302 strict metrics must not regress.

## Verification

- Documentation self-review ran in the same session.
- No code tests run for this design/plan log because implementation has not started.

## Result

Plan recorded.

## Follow-Up

- Execute the implementation plan.
- Create implementation, timing, and T302 non-regression logs after code changes and verification.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `pending`
