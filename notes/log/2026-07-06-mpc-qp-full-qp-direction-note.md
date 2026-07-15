# MPC QP Full-QP Direction Note

## Purpose

Record the user's design decision after viewer observation of discontinuous foot trajectories around small obstacles, difficult stairs, and box terrain.

## Stage

MPC-QP backend / trajectory quality direction.

## Related Todo

[T302v MPC QP safety-constrained backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Discussion-only note. No code changes and no tests were run.

## Decision

Do not continue solving the observed discontinuity by stacking more post-hoc safety repair. The next `mpc_qp` pass should move semantic/object collision, foothold keepout, height clearance, FK/body-leg safety approximations, and trajectory continuity into the QP formulation itself.

Existing repair layers should be treated as temporary scaffolding or explicit fallback until removed/demoted.

## Acceptance Direction

Future verification should include both hard safety and trajectory quality:

- semantic collision / touchdown-on-small / stance-on-small remain zero
- foot frame-to-frame jumps stay bounded
- joint frame-to-frame jumps stay bounded
- FK foot vs target foot readback error stays bounded
- swing arc and stance continuity remain visually stable on small obstacles, stairs, and box terrain
- IsaacLab viewer/probe evidence is rerun after repair demotion

## Result

Recorded only. Implementation remains open as Task 16 in the T302v branch page.

## Follow-up

Design and implement a full QP-centered `mpc_qp` path where repair is no longer the dominant behavior source.

## Git Refs

- Current Work Ref: discussion-only note on 2026-07-06
- Key Files: `notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md`
