# Joint MPC RTI Kinematic Task 09 Trajectory Scan

## Purpose

Solve the direct-state H30 block-pentadiagonal QP with a fixed H30/32 associative factor tree and preserve active box/velocity KKT semantics.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 9 of 16

## TDD Evidence

RED: focused collection failed because the old module had no conditional factor combine export and no direct-state trajectory scan module. Additional RED cases exposed two active-constraint defects during implementation: a box-fixed velocity destination lost the velocity equality, and a velocity edge with two box-fixed endpoints was not treated as redundant like the dense KKT reference.

GREEN:

- Focused Task 9 suite: `12 passed in 8.41s`.
- Monitored CUDA B1 compile: `1 passed, 11 deselected in 7.64s`, inside the `120s` timeout.
- Task 8 resume baseline: `6 passed, 17 deselected`.
- `py_compile` and `git diff --check`: exit `0` before final note updates.

## Metrics And Contract

- Conditional factor composition passes float64 associativity at `1e-10`.
- 30 real intervals append two identity/no-cost factors and use five explicit pair-combine levels.
- No generic PyTorch higher-order `associative_scan` remains in the scan modules.
- Dense active-KKT max absolute errors:
  - B=1: `6.93889390391e-17`
  - B=7: `1.11022302463e-16`
  - B=40: `1.38777878078e-16`
- CPU float32 parity passes the required `<2e-5` threshold.
- Active parity covers isolated boxes, isolated velocity edges, a velocity edge anchored by one box, two-box redundant velocity, and random feasible active components.
- The solve uses `y_k=[delta_z_(k-1),delta_z_k]` only as a 36-dimensional internal separator; public optimization variables remain `Z[B,31,18]`.

## Git Refs

- Baseline Ref: `3f5fb2d`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/solver/associative_scan.py`, `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_scan.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py`

## Follow-Up

Implement Task 10's exactly five parallel line-search candidates with finite/joint-position/joint-velocity filters and seven-loss-only selection.
