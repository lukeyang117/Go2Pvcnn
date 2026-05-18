# T302g Spec And Branch Todo Hardening

## Purpose

Record the documentation hardening requested after subagent review: fold the five review findings into the T302g spec and make the T302g branch page the detailed execution todo source of truth.

## Stage

Design/spec hardening and todo decomposition for independent MPC semantic RL train/play integration.

## Related Todo

- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
- Parent: [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Baseline Ref

- `446a875`

## Candidate Ref

- Working tree on top of `446a875`

## Key Files

- [../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md](../../docs/superpowers/specs/2026-05-18-mpc-semantic-rl-training-config-design.md)
- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
- [../todo.md](../todo.md)
- [index.md](index.md)
- [2026-05-18-1053-t302g-subagent-requirement-coverage-review.md](2026-05-18-1053-t302g-subagent-requirement-coverage-review.md)

## Changes

- Spec now states that future execution uses the T302g branch page child todo tree, not another plan file.
- Spec now requires explicit `planner_owned_reference_cache = True` and `use_batched_reference_trajectory = True` in both train and play config classes.
- Spec now requires scanner pose/yaw-aware map queries for reward sampling, with yaw/translation tests.
- Spec now defines swing/stance classification from current IsaacLab `contact_forces`, not planner contact imitation or MPC cache.
- Spec now requires 4096 timing to instantiate the new MPC semantic task and prefer RSL-RL rollout/collect-data timing, not only raw env stepping through the old viewer fixture.
- Spec now requires Gym registry assertions for both new ids and regression coverage for the old trajectory default path.
- T302g branch page now has child nodes `T302g.1` through `T302g.7` for config/registry, observation, reward, dirty replanning, timing, non-regression, and notes/log alignment.

## Verification

- Documentation diff reviewed manually.
- No code tests run; this pass only hardens spec/todo documents.

## Result

Pass for documentation hardening. Implementation remains pending.

## Follow-Up

- Execute `T302g.1` first from the branch page child todo tree.
- Create implementation, timing, and T302 non-regression logs during code work.

## Git Refs

- Last Feature Commit: `446a875`
- Last Verified Commit: documentation-only self-review
