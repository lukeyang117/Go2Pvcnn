# T300d MPC long replan foot motion and yaw display fix

- Time: 2026-05-11 16:55
- Stage: viewer/runtime MPC playback and long-horizon replan behavior
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline ref: `working tree on top of 130c635`
- Candidate ref: `working tree (uncommitted)`

## Purpose

Reproduce and fix two user-observed viewer issues:

- feet move briefly or not enough, then stop moving during longer MPC playback/replanning
- pressing `Q/E` should command yaw, but viewer status made it look like large x-axis rotation

## Reproduction

Procedure used `env_isaacsim` on `cuda:2`, following viewer runtime fixture and viewer-style rolling replanning:

- `forward` command over repeated replan cycles initially produced root progress but zero foot temporal displacement.
- `yaw_left` command produced correct planned yaw-only RPY in `wxyz`, but the viewer status printed the debug `xyzw` interpretation, where roll approached `pi`, creating the visual/log impression of a large x-axis rotation.

## Fix

- `Go2Pvcnn/extension/batch_mpc_planner/nominal.py`
  - Added command-relative nominal foot seed with gait phase:
    - stride-limited xy swing around the initial foot position
    - smooth swing interpolation
    - small swing-height bump
  - This preserves MPC residual optimization but breaks the static-foot symmetry that caused repeated replans to stop moving feet.
- `Go2Pvcnn/extension/batch_mpc_planner/config.py`
  - Added tunables:
    - `mpc_nominal_stride_scale`
    - `mpc_nominal_max_stride_m`
    - `mpc_nominal_swing_height_m`
  - Added `contact_schedule` loss config and overrides.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/contact.py`
  - Added `contact_schedule_tracking_loss`.
- `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
  - Wired `contact_schedule` into total loss.
- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - Viewer status now prints `actual_rpy_wxyz` as the primary orientation readout.
  - The old `xyzw` interpretation remains as `actual_rpy_xyzw_dbg` for debugging only.
- `Go2Pvcnn/tests/test_mpc_runtime_headless.py`
  - Added long viewer-style replan test requiring feet to keep moving across repeated replans.
  - Added yaw playback test requiring actual `wxyz` RPY to match plan and roll/pitch to stay near zero.

## Verification

1. `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q`
   - Result: `12 passed`
2. `MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k "viewer_style_replan_keeps_feet_moving or yaw_playback_wxyz_rpy_matches_plan" -q`
   - Result: `2 passed`
3. `MPC_TEST_DEVICE=cuda:2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -q`
   - Result: `7 passed, 1 skipped`

## Conclusion

The long-replan no-foot-motion issue was caused by a static foot nominal that left the optimizer at a symmetric static-foot solution. The `Q/E` x-axis rotation issue was primarily a viewer status readout convention issue: `wxyz` matches the planned yaw-only orientation, while the debug `xyzw` interpretation can display a roll near `pi`.
