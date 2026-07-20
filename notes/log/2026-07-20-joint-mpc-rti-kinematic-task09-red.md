# Joint MPC RTI Kinematic Task 09 RED

## Purpose

Prove that the H30/32 associative trajectory solver contract is absent before implementation.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 9 of 16, RED

## Result

The focused test collection failed because `combine_conditional_value_factors` was not exported by the old generic scan module and `trajectory_scan.py` did not exist. This is the expected missing-feature failure.

## Git Refs

- Baseline Ref: `3f5fb2d`
- Candidate Ref: uncommitted Task 9 RED test
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_scan.py`

## Follow-Up

Implement conditional factor composition, H30-to-H32 padding, the explicit five-level tree, active local parameterization, and full-node recovery.
