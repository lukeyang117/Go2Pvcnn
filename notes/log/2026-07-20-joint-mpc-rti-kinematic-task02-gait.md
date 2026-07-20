# Joint MPC RTI Kinematic Task 02 Gait

## Purpose

Replace the adjustable H16 contact helper with the approved fixed 24-frame diagonal trot tensor schedule.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 2 of 16

## TDD Evidence

RED: all three new tests failed because the historical function required explicit `batch`, `device`, and adjustable `half_cycle_steps` arguments and returned only a contact tensor.

GREEN command:

```bash
PYTHONPATH=Go2Pvcnn \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py -q
```

Result: `10 passed in 2.62s`.

## Contract

- Input phase is `[B]`; output phase, swing, stance, and swing tau are `[B,H+1,4]`.
- FL/RR and FR/RL are diagonal pairs and exact complements.
- Every leg has exactly 12 swing nodes and 12 stance nodes per 24-node period.
- The implementation contains no recovery or swing-extension state.
- Existing FK and complete link/Jacobian tests remain green.

## Git Refs

- Baseline Ref: `3a1922c`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_fixed_gait_schedule.py`

## Follow-Up

Add vectorized analytic IK while preserving FK geometry.
