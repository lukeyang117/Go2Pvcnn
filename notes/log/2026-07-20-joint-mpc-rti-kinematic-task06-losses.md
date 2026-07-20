# Joint MPC RTI Kinematic Task 06 Losses

## Purpose

Replace the control/recovery-heavy objective with exactly seven pure-state trajectory losses shared by nonlinear scoring and later GGN linearization.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 6 of 16

## TDD Evidence

RED: collection failed because the old objective had no `LossContext` or seven-state trajectory API.

Focused GREEN: `8 passed in 3.13s`.

Combined Tasks 1-6 CPU result: `45 passed, 13 skipped in 5.43s`.

## Contract

- Exact ordered keys: `command`, `step`, `contact`, `swing_speed`, `terrain`, `posture`, `smooth`.
- Every loss consumes `state[B,31,18]` and returns one value per environment.
- Every loss exposes a fixed-shape residual; total loss is the weighted half squared norm of the same residuals used by GGN.
- Command, contact, swing speed, and smoothness derive velocity/differences only from adjacent state nodes.
- Swing speed penalizes swing-foot XY progress that is not greater than root XY progress plus the configured margin.
- Terrain performs one packed query of 41 full-body points per node and uses only soft occupancy, propagated height, and continuous virtual walls in the optimizer path.
- Terrain source contains no raw semantic-ID access or class branch; stance-on-small remains a continuous occupancy residual under the terrain parent loss.
- Smoothness contains first and second state differences.
- Old control, terminal, recovery, startup, and dozens-of-key loss tests were replaced by the seven-key contract and finite-gradient/full-body tests.

## Git Refs

- Baseline Ref: `a54ad42`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/losses/`, `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_losses.py`

## Follow-Up

Linearize the weighted seven-loss residual directly in Z and assemble the block-pentadiagonal trajectory QP.
