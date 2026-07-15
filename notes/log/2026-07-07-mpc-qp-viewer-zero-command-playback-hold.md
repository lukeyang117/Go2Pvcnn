# 2026-07-07 MPC QP Viewer Zero-Command Playback Hold

## Purpose

Fix the user's live viewer report that `mpc_qp` still jitters after releasing velocity keys / returning command to zero.

## Stage

MPC-QP viewer runtime direct playback.

## Related Todo

[T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Root Cause

The earlier manager-level zero-command hold was correct for manager-owned reference caches, but the interactive viewer does not display through that cache. The viewer main loop directly calls `_plan_viewer_trajectory()` and `_viewer_direct_playback_step()`.

In the moving-to-zero case, `active_cmd.values` could become zero while an older moving `ViewerTrajectoryResult` was still being played. Because the old result reached `_viewer_direct_playback_step()`, the visible robot kept moving for several frames even though the printed command was already zero.

## Change

- Added `_viewer_should_hold_mpc_qp_zero_command()` for `mpc_qp` only.
- Extended `_viewer_plan_has_motion()` to check root, quat, joints, and feet.
- Added a viewer playback guard immediately before `_viewer_direct_playback_step()`:
  - if backend is `mpc_qp`
  - and the current command is zero
  - and the current result still has motion
  - replace the result with `_viewer_hold_result_from_previous_final_frame(result)`
  - reset `playback_frame` and idle-debug sample state
- Existing `mpc` behavior is unchanged.

This is a viewer runtime hold. It is not a QP loss change, not a candidate endpoint/search method, and not a hard repair.

## Verification

Focused unit tests:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_viewer_zero_command_hold_uses_previous_result_final_frame Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_viewer_zero_command_forces_hold_when_previous_result_still_moves Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_viewer_idle_debug_delta_formats_root_and_foot_motion -q
python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_qp_backend.py
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Results:

- focused viewer tests: `3 passed`
- pycompile: pass
- full QP unit suite: `69 passed`

Real IsaacSim scripted moving-to-zero viewer run:

```bash
CUDA_VISIBLE_DEVICES=3 timeout 150s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --qp-iterations 3 --scripted-command '0.45 0 0' --scripted-command-cycles 2 --idle-debug --idle-debug-stride 5
```

Before the playback guard, the same scripted sequence reproduced the bad zero-command playback:

- zero-command cycle frame `5`: `foot_delta_max_m≈0.18604`, `root_delta_m≈0.03495`
- zero-command cycle frame `10`: `foot_delta_max_m≈0.13465`, `root_delta_m≈0.08941`

After the playback guard:

- the first zero-command rows switch immediately to static hold
- subsequent zero-command rows report `root_delta_m=0`
- subsequent zero-command rows report `foot_delta_max_m=0`
- no old moving frames are played after command becomes zero

Operational note:

- One rerun initially failed with CUDA OOM / illegal memory access because a previous viewer process I started was still occupying GPU 3. After terminating that stale process, the real run completed and produced the after metrics above.

## Conclusion

The visible fast jitter on command release was a viewer direct-playback stale-result problem. `mpc_qp` now holds the previous planned final frame before writing any further zero-command frames to the robot.

## Follow-Up

- The user's exact livestream command can now be rerun interactively; this pass verified the same viewer loop headless with scripted command transition and `qp_iterations=3`.
- This does not address separate non-idle flat-small trajectory quality issues.

## Git Refs

- Baseline Ref: dirty worktree after manager-level zero-command hold.
- Candidate Ref: dirty worktree after viewer playback zero-command hold.
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
