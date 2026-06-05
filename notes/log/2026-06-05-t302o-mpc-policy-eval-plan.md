# T302o MPC Policy Eval Plan

## Purpose

Record the implementation plan for the MPC policy evaluation script requested by the user.

## Stage

MPC semantic policy evaluation planning.

## Related Todo

- [T302o](../todo/T302o-mpc-policy-eval-plan.md)

## Git Refs

- Baseline Ref: `f46eab8`
- Candidate Ref: working tree after T302o todo updates
- Current Work Ref: `costmap-teacher-ablation`

## Key Files

- [../../docs/superpowers/specs/2026-06-05-mpc-policy-eval-design.html](../../docs/superpowers/specs/2026-06-05-mpc-policy-eval-design.html)
- [../todo.md](../todo.md)
- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)
- [index.md](index.md)

## Command / Procedure

No code or runtime test was executed. The procedure was:

1. Update [../todo.md](../todo.md) to make T302o the active front.
2. Create [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md) as the detailed implementation plan.
3. Update [index.md](index.md) with this planning log.

## Input Conditions

- User approved the Python-only evaluation script direction.
- User requested no shell launcher.
- User specified `max_steps` completes one round and `num_rounds` controls how many rounds run.
- User specified small collision rate denominator is environment count, not step count.
- User requested livestream mode to send the same velocity command to policy and MPC and visualize MPC foot trajectories.

## Key Metrics

- Plan file created: `1`
- Implementation tasks: `7`
- Runtime tests executed: `0`
- Code files changed: `0`

## Result

Plan recorded.

## Conclusion

T302o is now the active implementation front in the notes dashboard. Implementation has not started. The next step is Task 1 in [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md).

## Follow-up

- Execute T302o task-by-task using the required implementation workflow.
- Preserve `scripts/play.py` no-MPC behavior.
