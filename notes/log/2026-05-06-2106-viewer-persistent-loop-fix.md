# 2026-05-06 21:06 Viewer Persistent Loop Fix

## Scope

- Todo: [T100/T111 viewer livestream black-screen / GLFW disconnect triage](../todo/T100-batched-together-planner-gpu-migration.md#t111-viewer-livestream-black-screen--glfw-disconnect-triage)
- Stage: `Go2Pvcnn/extension/viz/go2_foostep_planner.py` remote WebRTC viewer lifecycle
- Environment: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`

## Context

After the WebRTC endpoint and IsaacLab livestream-extension fixes, the user reported that the local visualization window now appears, but the server process exits shortly after printing:

- `Creating window for environment.`
- `[INFO]: Completed setting up the environment...`
- `[Viewer] Attached together trajectory manager`
- terminal teleop help
- `GLFW initialization failed`
- `add_menu_items ... cannot change delegate`

The user requirement is to keep the viewer alive until manual `Ctrl-C`.

## Change

- Changed the viewer main loop from `while simulation_app.is_running():` to `while True:`.
- Added an explicit `KeyboardInterrupt` handler that prints `[Viewer] Ctrl-C received; shutting down.` and then runs the existing cleanup path.
- Did not add a legacy-mode flag; the requested behavior is now the direct/default behavior.

## Verification

- AST check confirmed no `while simulation_app.is_running()` loop remains in `go2_foostep_planner.py`.
- `python -m py_compile` passed for:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/tests/test_viz_playback.py`
  - `/mnt/mydisk/lhy/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py`
- `git diff --check` passed for touched project files and IsaacLab `app_launcher.py`.

## Caveats

- Full interactive browser-side runtime duration remains a manual confirmation item because running another WebRTC viewer from the agent could steal the user's client/port.
- The existing focused pytest target is blocked before test collection by repository-local `Go2Pvcnn/tests/conftest.py` importing missing `scripts.go2fp`; this is an existing test-environment issue, not caused by the loop change.
