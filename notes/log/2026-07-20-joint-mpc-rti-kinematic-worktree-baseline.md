# Joint MPC RTI Kinematic Worktree Baseline

## Purpose

Establish a clean isolated baseline before executing the approved pure-kinematic H30 replacement plan.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Worktree: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim-joint-mpc-kinematic`
- Branch: `work/joint-mpc-kinematic`

## Procedure

Focused existing contract, kinematics, loss, solver, and terrain tests were run with a 180-second outer timeout. The same suite was then rerun with CUDA hidden to separate host contracts from current GPU availability.

```bash
CUDA_VISIBLE_DEVICES='' timeout 180s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_contracts.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_kinematics_gait.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_losses.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_solver.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py -q
```

## Result

- CPU baseline: `62 passed, 13 skipped in 13.33s`.
- GPU-visible attempt: the same 62 host tests passed; 13 CUDA terrain tests failed at allocation with `CUDA error: out of memory`.
- Worktree status before implementation was clean at `9168f1d`.
- The original `joint_mpc` checkout and its uncommitted adaptive-contact work were not modified.

## Conclusion

The isolated branch is a valid host baseline. CUDA verification remains pending until it runs under the approved monitored runner on a GPU with sufficient free memory; the allocation failures are environment evidence, not accepted test regressions.

## Git Refs

- Baseline Ref: `9168f1d`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/`, `Go2Pvcnn/tests/joint_mpc_rti/`

## Follow-Up

Execute Task 1 of the approved implementation plan with RED contract tests.
