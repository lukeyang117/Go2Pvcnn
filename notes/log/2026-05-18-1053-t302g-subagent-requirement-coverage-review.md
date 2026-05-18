# T302g Subagent Requirement Coverage Review

## Purpose

Record the subagent review of whether the T302g MPC semantic RL training requirements are fully covered by the written spec and implementation plan.

## Stage

Design/plan review for independent MPC semantic RL train/play configuration before implementation.

## Related Todo

- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
- Parent: [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Baseline Ref

- `446a875`

## Candidate Ref

- `446a875`

## Key Files

- [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
- [../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md](../../docs/superpowers/plans/2026-05-18-mpc-semantic-rl-training-config.md)
- [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)

## Procedure

- Used the existing review subagent result requested by the user.
- Verified the review findings against the plan/spec and runtime fixture source.

## Covered Requirements

- New T302 child branch/page exists and points to the T302 parent.
- New task config is planned in a new Python file.
- Existing `teacher_elevation_trajectory` / together defaults are intended to stay unchanged.
- Train/play experiment choices and new Gym ids are specified.
- `semantic_height_scanner` at `0.01m` is specified for MPC/reward high-resolution use.
- CNN input is specified as `2 x 16 x 16`, with priority semantic pooling.
- MPC imitation is foot-only.
- Dirty-subset replanning is specified conceptually and has planned tests.
- T302 non-regression baseline is tied to the strict `17/17` metric log.

## Findings

1. P0: The 4096 timing test is currently planned through the old viewer/play fixture, which hardcodes `Isaac-Teacher-Elevation-Trajectory-Go2-Play-v0`. It may not instantiate the new MPC semantic train config or actual collect-data path.
2. P1: The new config block in the plan does not explicitly set `planner_owned_reference_cache = True` and `use_batched_reference_trajectory = True`, although the spec requires them.
3. P1: `swing_leg_collision_reward` is planned as an equal average over leg bodies; it does not yet implement or test stronger swing-vs-stance weighting.
4. P1: Reward terrain sampling maps world `xy` to fixed scanner ranges and does not yet account for scanner pose/yaw alignment.
5. P2: Registry coverage needs explicit `gym.spec(...)` assertions for both new Gym ids plus a regression assertion that existing trajectory mappings/defaults remain unchanged.

## Result

Partial pass. The main requirements are captured, but the spec and T302g branch-page child todo tree need hardening before implementation starts.

## Follow-Up

- Patch the spec and T302g branch page to address the five findings before coding.
- Do not create a new implementation plan unless the user explicitly asks for one.
- Then execute the T302g branch child todos and create separate implementation, timing, and non-regression logs.

## Git Refs

- Last Feature Commit: `446a875`
- Last Verified Commit: `446a875`
