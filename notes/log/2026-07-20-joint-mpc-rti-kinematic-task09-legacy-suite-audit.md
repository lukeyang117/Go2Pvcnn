# Joint MPC RTI Kinematic Task 09 Legacy Suite Audit

## Purpose

Audit the broader historical solver test file after Task 9 without restoring APIs superseded by the approved pure-kinematic design.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Related plan tasks: 10 and 11 migration cleanup

## Result

The combined trajectory-QP, trajectory-scan, and full historical solver command reported `27 passed, 7 failed`.

All seven failures are stale old-architecture contracts already scheduled for deletion or replacement in Tasks 10/11:

- control-based `shift_warm_start`
- `coupled_state_riccati` legacy config
- control-dynamics Jacobians
- old planner imports through `rollout_objective` and `command_losses`
- old root-control warm start and old clearance-gradient planner helpers

No failure came from Task 9 scan tests. The relevant affine scan regression passed separately.

## Conclusion

Do not restore these APIs because doing so would violate the frozen direct-Z/no-control architecture. Keep this as an explicit migration debt item until Tasks 10/11 replace and delete the stale tests and old planner path.

## Git Refs

- Baseline Ref: `3f5fb2d`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/test_solver.py`, `Go2Pvcnn/extension/joint_mpc_rti/planner.py`

## Follow-Up

Task 10 removes line-search tests that require old safety ranking; Task 11 removes the remaining old control/dynamics/planner tests after the new one-RTI pipeline is wired.
