# Joint MPC RTI Kinematic Task 07 Direct-Z QP

## Purpose

Linearize the exact weighted seven-loss residual directly in `Z[B,31,18]` and retain the block-pentadiagonal GGN bands.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 7 of 16

## TDD Evidence

RED: the historical linearization module exposed only control/dynamics Jacobians and no direct trajectory QP.

GREEN result: `3 passed in 4.25s`.

## Contract

- Per-environment residual Jacobians are formed with `vmap(jacrev)`; no cross-environment dense Jacobian is materialized.
- GGN stores only `[B,31,18,18]` diagonal, `[B,30,18,18]` first off-diagonal, and `[B,29,18,18]` second off-diagonal blocks.
- Dense test reference exactly matches nonlinear autograd gradient and weighted residual `J^T J + regularization` within the approved tolerances.
- Bounds merge trust region and joint position limits into one state box.
- `delta z0` lower and upper bounds are exactly zero.
- Joint velocity bounds are edge constraints on `delta q[k+1]-delta q[k]` with shape `[B,30,12]`.
- No stance, collision, startup, recovery, or semantic constraint row is present.

## Git Refs

- Baseline Ref: `949d0e4`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/solver/linearization.py`, `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py`

## Follow-Up

Select fixed-shape active box/velocity constraints and perform exactly two unrolled refinements.
