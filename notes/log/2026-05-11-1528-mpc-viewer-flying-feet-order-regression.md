# T300d MPC Viewer Flying-Feet Order Regression

- timestamp: 2026-05-11 15:28 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass

## Purpose

Investigate user-reported MPC viewer symptom: foot endpoints appear to "fly" during playback.

## Stage

`go2_foostep_planner.py` MPC viewer runtime path (`_mpc_state_from_env` -> `_plan_viewer_trajectory` -> `_apply_direct_playback_to_robot`).

## Root Cause

MPC viewer state construction used robot-native joint order directly in `MpcRobotState.joint_angles`, while playback path interprets planned joints using planner order.

- input path: `robot.data.joint_pos` (robot order)
- playback path: `_joint_pos_planner_to_robot(...)` (expects planner order)
- effect: joint sequence mismatch corrupts leg pose during playback, producing large visual foot drift.

## Changes

- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - in `_mpc_state_from_env`, convert joint positions with `_joint_pos_robot_to_planner(...)` before constructing `MpcRobotState`.

- [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - added `test_mpc_runtime_viewer_playback_kinematics_consistency` regression:
    - strict `root/joint` playback equality checks
    - bounded foot-position norm error guardrail to catch order-regression reintroduction.

## Verification

1. `env_isaacsim` headless runtime script on `cuda:2` (viewer fixture path):
   - `standstill`: `root_max=0.000000`, `joint_max=0.000000`, `foot_max=0.041151`
   - `forward`: `root_max=0.000000`, `joint_max=0.000000`, `foot_max=0.056443`
2. exact viewer runtime command path (`--planner-backend mpc`, headless) reached MPC plan/playback loop and printed:
   - `[Viewer][ActualKinematics] ... joint_err_max=0.000000 ... foot_err_max=0.038480`
   - no module import crash; no decimeter-level flying-feet divergence.
3. targeted regression test:
   - `MPC_TEST_DEVICE=cuda:2 python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k playback_kinematics_consistency -q` -> `1 passed`
4. cross-GPU replay check on `cuda:3`:
   - `standstill`: `foot_norm_max=0.041151`
   - `forward`: `foot_norm_max=0.069852`
   - `root_max/joint_max` remained `0.000000`.

## Conclusion

The large flying-feet artifact was caused by MPC state joint-order mismatch in viewer state construction. After planner-order conversion, playback root/joint alignment is exact and foot drift is reduced to centimeter-level residuals with stable headless behavior on both `cuda:2` and `cuda:3`.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: working tree with MPC viewer joint-order fix + playback regression test
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
