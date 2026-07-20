# Joint MPC RTI Kinematic Task 08 Resume Baseline

## Purpose

Reconfirm the fixed active-bound and compile-budget baseline before starting the H30/32 associative trajectory solver.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 8 of 16 resume verification

## Command

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_trajectory_qp.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py \
  -k 'active or compile_budget or trajectory_qp' -q
```

## Result

- `6 passed, 17 deselected in 6.18s`
- Exit code `0`.

## Conclusion

Task 8 remains GREEN. Task 9 can use the current `TrajectoryQp`, `ActiveConstraints`, and dense active-KKT reference as its parity contract.

## Git Refs

- Baseline Ref: `3f5fb2d`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/solver/trajectory_qp.py`, `Go2Pvcnn/extension/joint_mpc_rti/solver/primal_dual_ilqr.py`

## Follow-Up

Implement Task 9 with RED-first associativity, padding, source-contract, and dense parity tests.
