# T300d MPC Viewer Forward Static-Joint IK Fix

- timestamp: 2026-05-11 15:43 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass

## Purpose

Fix user-reported MPC viewer runtime issue: command drives root motion, but all four legs appear static (joint trajectory not changing over horizon).

## Stage

`batch_mpc_planner` planning output contract used by viewer/runtime playback:

- planner output generation: `extension/batch_mpc_planner/planner.py`
- viewer playback input: `result.joint_angles[B,T,12]`

## Root Cause

`plan_segment()` produced `joint_angles` by repeating initial state joints for all frames:

- `joint_seed.unsqueeze(1).expand(B,T,12)` was returned directly
- therefore `joint_angles` stayed time-constant even when `root_pos` changed
- viewer then replayed a moving base with effectively static leg joints.

## Changes

- [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - added GPU batched IK helper `solve_joint_angles_from_trajectory(root_pos, root_rpy, foot_pos_w)` for `[B,T]` trajectory solve.

- [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - keep seed joints only for current optimization-loss path input.
  - replace result joint sequence with IK-solved trajectory from optimized `decoded.root_pos/root_rpy/foot_pos`.

- [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - added regression `test_mpc_runtime_forward_plan_has_time_varying_joint_angles`:
    - forward plan must move root (`dx > 0.05`)
    - foot trajectory must vary (`foot_tspan > 0.01`)
    - joint trajectory must vary over time (`joint_tspan_max > 1e-3`).

## Verification

All verification on `env_isaacsim` + `cuda:2`, using viewer-compatible IsaacLab headless runtime path.

1. Red test before fix:
   - `MPC_TEST_DEVICE=cuda:2 python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k time_varying_joint_angles -q`
   - result: `F`

2. Green test after fix:
   - same command
   - result: `.`

3. Focused regression subset:
   - `MPC_TEST_DEVICE=cuda:2 python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k "plan_case_headless_smoke or playback_kinematics_consistency or time_varying_joint_angles" -q`
   - result: `...`

4. Runtime metric probe (headless fixture script, forward command):
   - `ROOT_DELTA_ABS [0.2917, 0.0, 0.0]`
   - `JOINT_TSPAN_MAX 0.9467` (previously `0.0`)
   - `JOINT_TSPAN_MEAN 0.4528` (previously `0.0`)

5. Syntax sanity:
   - `python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/kinematics.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_mpc_runtime_headless.py`
   - result: pass

## Conclusion

The static-leg issue was caused by planner outputting horizon-constant joints. MPC now outputs time-varying joint trajectories solved from optimized root+foot trajectories, and headless viewer/runtime path confirms nonzero joint time span under forward command.

## Follow-Up

- Current optimization-loss path still uses seed-joint input during residual optimization loop. If needed, a follow-up can wire per-iteration IK joints into kinematics loss for tighter optimization-consistency.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: working tree with MPC IK output fix
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
