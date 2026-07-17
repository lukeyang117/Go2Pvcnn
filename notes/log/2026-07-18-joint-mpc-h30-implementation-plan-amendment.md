# Joint MPC H30 Implementation Plan Amendment

## Purpose

Amend the existing 2026-07-17 root-joint coupled gait plan after approval of the H30 adaptive-contact/root-assist design.

## Stage

T302v.8 planning. Stage A starts next; Stage B and final joint verification remain blocked by sequence.

## Related Todo

- [T302v joint MPC RTI GPU](../todo/T302v-joint-mpc-rti-gpu.md)
- [Investigation dashboard](../todo.md)

## Result

The original plan at `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md` now:

- fixes production and acceptance to H30 with 15-frame stance/swing;
- supersedes H16-H50 selection and variable `H_selected` instructions;
- adds TDD tasks for per-leg contact/recovery, safe touchdown, bounded root lateral/RPY assistance, all-body small-obstacle LQ directions, complete JointMetrics and the 275-command matrix;
- requires Stage A to pass before Stage B starts;
- fixes Stage B to realistic idle-GPU `1024 x H30 x 1000 <=5000ms`;
- requires a fresh same-candidate Stage A + Stage B rerun before completion.

## Verification

- Plan placeholder/diff check passed.
- Legacy Tasks 8-14 are explicitly marked as superseded audit history.
- No planner implementation changed in this planning commit.

## Follow-up

Execute only fixed-H30 Stage A first. Do not start Stage B until the complete latest Stage A report is green.

## Git Refs

- Baseline Ref: `06084ce`
- Candidate Ref: `joint_mpc` working tree
- Key Files: `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md`
