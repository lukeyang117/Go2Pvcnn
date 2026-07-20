# Joint MPC RTI Kinematic Task 11 RED

## Purpose

Prove the production planner had not yet been routed through the approved pure-kinematic one-RTI pipeline.

## Result

Three of four new pipeline tests failed at old planner import time because `planner.py` still depended on the historical control rollout objective and removed `command_losses` API. The compact solver-state contract test already passed.

## Git Refs

- Baseline Ref: `0ef672a`
- Candidate Ref: uncommitted Task 11 RED tests
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py`, `Go2Pvcnn/extension/joint_mpc_rti/planner.py`

## Follow-Up

Replace planner and SQP orchestration, update runtime ABI, and delete old control/recovery/projection modules and tests.
