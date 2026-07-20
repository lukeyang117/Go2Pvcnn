# Joint MPC RTI Kinematic Task 01 Contract

## Purpose

Freeze the approved pure-kinematic production contract and remove independent control/recovery data from the solver-state ABI.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 1 of 16

## TDD Evidence

RED failed in the expected three places: old horizon `16` instead of `30`, no seven-weight `weights()` interface, and old solver-state fields `state/control/dual/previous_control/gait_phase/stance_anchor_w`.

GREEN command:

```bash
PYTHONPATH=Go2Pvcnn \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_new_contract.py -q
```

Result: `3 passed in 1.56s`.

## Contract

- `H=30`, `dt=0.02`, one SQP/RTI iteration.
- Gait period `24`, swing `12`, stance `12`.
- Line-search alpha tuple `(1.0, 0.5, 0.25, 0.125, 0.0)`.
- Exactly seven top-level loss weights.
- Solver state fields are exactly `trajectory`, `gait_phase`, and `valid`.
- Trajectory stores derived velocity diagnostics rather than an optimized control sequence.

## Git Refs

- Baseline Ref: `9168f1d`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/config.py`, `Go2Pvcnn/extension/joint_mpc_rti/types.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_new_contract.py`

## Follow-Up

Implement the fixed broadcasted 24-frame trot schedule in Task 2.
