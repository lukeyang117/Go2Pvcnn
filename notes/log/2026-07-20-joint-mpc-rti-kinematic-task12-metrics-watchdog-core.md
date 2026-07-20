# Joint MPC RTI Kinematic Task 12 Metrics And Watchdog Core

## Purpose

Start the shared flat/small metric applicability layer and process-group watchdog without launching CUDA or Isaac Lab.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 12 of 16, active checkpoint

## TDD Evidence

- RED: `6 failed, 2 passed`; missing applicability APIs, 275-command matrix, acceptance registry, and monitored runner.
- Current GREEN: `8 passed in 1.82s`.

## Implemented Contract

- Formal signed command product contains exactly 275 combinations.
- `applicable_metrics("flat")` is a strict subset of `applicable_metrics("small")`.
- Flat small-specific metrics carry explicit `no small obstacle in flat scenario` N/A metadata.
- Each metric result has value, numerator, denominator, valid count, applicability, N/A reason, threshold, pass, and worst-case fields.
- Timeout supervision launches a new process session and terminates only the child process group with TERM then KILL grace handling.

## Still Open

- Complete the approved trace fields and threshold calculations.
- Add heartbeat/resource monitoring for process-tree RSS, ptxas RSS, available-memory/swap deltas, and selected-GPU progress.
- Implement the complete unified acceptance CLI/report schema and per-cell progress.
- Map old heavy-probe coverage before deleting silent routes.

## Git Refs

- Baseline Ref: `156a6c0`
- Candidate Ref: uncommitted Task 12 checkpoint
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`, `Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py`, `Go2Pvcnn/tests/joint_mpc_rti/scenario_matrix.py`
