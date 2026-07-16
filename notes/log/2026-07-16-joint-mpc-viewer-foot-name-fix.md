# Joint MPC Viewer Foot Name Fix

- Purpose: fix the viewer crash in `_read_actual_kinematic_state()` caused by an undefined foot-name normalization helper.
- Stage: viewer diagnostics and shared Isaac/planner name conversion.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `5a94255`.
- Candidate Ref: `b99cda0`.
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/integration/joint_order.py`, `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, `Go2Pvcnn/tests/test_viewer_reset.py`.

## Root Cause

The viewer foot-body reorder path called `_normalize_joint_name`, but that private helper existed only inside `joint_order.py` and was not imported into the viewer. The path was reached after direct playback when viewer diagnostics read actual foot positions.

## Change

- Promoted the shared helper to public `normalize_articulation_name()`.
- Reused it for both joint-order indices and foot-body order conversion.
- Added a namespaced/path-qualified body-name regression test.

## Verification

- RED: focused regression failed with the same `NameError` at `go2_foostep_planner.py:426`.
- GREEN: focused regression `1 passed`.
- Viewer plus complete joint MPC suite: `133 passed in 35.25s`.
- Real one-environment Isaac headless call to `_read_actual_kinematic_state()`: joint shape `[1,12]`, foot shape `[1,4,3]`, all finite, exit `0`.
- `py_compile` and `git diff --check`: exit `0`.

## Conclusion

The reported viewer traceback is fixed at the shared name-conversion boundary. No planner loss, gait, stance, command, or playback behavior changed.
