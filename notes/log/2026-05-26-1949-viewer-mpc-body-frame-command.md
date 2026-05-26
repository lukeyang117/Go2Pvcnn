# Viewer MPC Body-Frame Command

## Purpose

Make viewer MPC planning interpret teleop/scripted `vx vy yaw_rate` as root/body-frame linear velocity before sending the command into the MPC planner.

## Stage

`extension/viz` viewer runtime boundary for `extension/batch_mpc_planner`.

## Related Todo

[../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Command / Procedure

TDD red:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_rotates_mpc_body_frame_command_by_root_yaw -q
```

Result: failed with missing helper `_viewer_mpc_world_command_from_root_frame`.

Green verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_rotates_mpc_body_frame_command_by_root_yaw -q
```

Result: `1 passed`.

Focused viewer regression:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viewer_reset.py -q
```

Result: `14 passed`.

Compile check:

```bash
python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py
```

Result: exit `0`.

## Input Conditions

The viewer command UI and `--scripted-command` describe body-frame/root-frame command values. Current MPC parametric decoding treats nonzero command XY as a world-frame direction, so a root yaw offset needs a boundary conversion before calling MPC.

## Key Metrics

- Root yaw `pi/2`, body command `(vx=0.4, vy=0.0, yaw=0.2)` converts to world command `(vx=0.0, vy=0.4, yaw=0.2)`.
- Focused tests: `14 passed`.

## Result

Pass locally. Added a viewer helper to rotate MPC linear command XY by current `state.root_rpy[:, 2]`; yaw-rate is preserved. The conversion is applied only in `_plan_viewer_trajectory()` for `backend="mpc"`.

## Conclusion

Viewer MPC now preserves the user-facing root-frame command contract without changing the MPC planner's internal command convention or non-MPC viewer backends.

## Follow-up

Real IsaacLab viewer visual confirmation remains unrun in this pass.

## Git Refs

- Baseline Ref: working tree on top of `1b799cd`
- Candidate Ref: working tree on top of `1b799cd`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
