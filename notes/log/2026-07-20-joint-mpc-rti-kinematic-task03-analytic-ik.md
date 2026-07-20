# Joint MPC RTI Kinematic Task 03 Analytic IK

## Purpose

Add closed-form batched Go2 IK for the one-call nominal builder while preserving complete FK and Jacobian geometry.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 3 of 16

## TDD Evidence

RED: test collection failed with `ModuleNotFoundError` for the new `model.analytic_ik` module.

GREEN command:

```bash
PYTHONPATH=Go2Pvcnn \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_analytic_ik.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -q
```

Result: `10 passed in 2.77s`.

## Contract

- IK accepts root pose `[...,3]` and four world foot targets `[...,4,3]`.
- Output angles are `[...,4,3]`; reachability is `[...,4]`.
- The implementation has no Python time or leg loop.
- Unreachable targets remain finite and are reported; output is not clipped to joint limits.
- Public FK/foot-FK now preserve arbitrary leading batch dimensions.
- Existing foot, knee, calf, thigh, body, and complete Jacobian tests remain green.

## Git Refs

- Baseline Ref: `eceb87a`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/model/analytic_ik.py`, `Go2Pvcnn/extension/joint_mpc_rti/model/go2_kinematics.py`

## Follow-Up

Build cold and shifted rolling nominal trajectories in one tensor call.
