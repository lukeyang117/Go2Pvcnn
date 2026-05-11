# Viewer WebRTC Public IP Fix

- Time: 2026-05-06 20:11 +0800
- Todo: [T100/T111](../todo/T100-batched-together-planner-gpu-migration.md#t111-viewer-livestream-black-screen--glfw-disconnect-triage)
- Stage: `extension/viz` remote WebRTC viewer startup
- Result: pass with scoped caveat

## Context

The user reported that the remote viewer command

```bash
python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --device cuda:0 --num_envs 1 --terrain task --planner-backend together
```

started but showed no visual image in the browser, with `carb.livestream-rtc.plugin` `nvstPushStreamData` errors, `NVST_CCE_DISCONNECTED`, connection-count mismatch warnings, and later `GLFW initialization failed`.

Earlier triage showed the viewer playback path already writes robot state, scene data, renders, and updates the scene, so the next question was whether the WebRTC endpoint was being advertised correctly for a remote server.

## Root Cause

IsaacLab's `AppLauncher` defaults the WebRTC public endpoint to `PUBLIC_IP`, and falls back to `127.0.0.1` when `PUBLIC_IP` is not set. On a remote server this can make the browser connect/disconnect without receiving a useful stream even though the planner and simulation are running.

`GLFW initialization failed` still appears in headless WebRTC logs, but the scripted smoke tests continue into `Streaming server started`, `[Viewer][Plan]`, and `[Viewer][Playback]`, so it is not the primary server-side blocker for this case.

## Changes

- Added `--webrtc-public-ip` to [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py).
- Added `--webrtc-port` with default `49100`.
- Added `--no-webrtc-auto-public-ip`.
- For `--livestream 2`, the viewer now sets `PUBLIC_IP` before `AppLauncher` starts, using this priority:
  - explicit `--webrtc-public-ip`
  - existing `PUBLIC_IP`
  - SSH server IP parsed from `SSH_CONNECTION`, unless disabled
- Updated viewer tests for CLI public IP, SSH inference, and non-default port Kit args.
- Updated the human command guide with the remote WebRTC public IP contract.

## Verification

- Environment import smoke in `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim` passed for torch/CUDA and IsaacLab imports.
- GPU check found unrelated PID `581038` occupying GPU 0/1 memory, so real viewer smokes used `cuda:2`.
- `--livestream 0` scripted smoke on `cuda:2` reached `[Viewer][Plan] backend=together`, `[Viewer][Playback] path=render+scene_sync`, and actual/plan kinematics matched with near-zero error.
- `--livestream 2` scripted smoke on `cuda:2` reached `Streaming server started`, `[Viewer][Plan]`, and `[Viewer][Playback]`; this reproduced non-fatal `GLFW initialization failed` warnings but not a planner stall.
- `--livestream 2 --webrtc-public-ip 172.31.179.75` scripted smoke printed `WebRTC public endpoint PUBLIC_IP=172.31.179.75 source=cli`, reached `Streaming server started`, `[Viewer][Plan]`, `[Viewer][Playback]`, and `[Viewer][ActualBase]`.
- Direct `_prepare_runtime_args()` checks passed for CLI public IP, SSH inference, and port override.
- `py_compile` passed for the viewer and updated viewer tests.

## Caveats

- Full pytest collection is blocked in this worktree because `raw/kinematic_footsteps` is empty, causing `ModuleNotFoundError: No module named 'scripts.go2fp'` from `Go2Pvcnn/tests/conftest.py`.
- Browser-side visual confirmation from the user's client is still unverified. If the browser still shows black, the next likely blockers are network/firewall/tunnel access to WebRTC port `49100`, browser-side ICE/WebRTC failure, or using the wrong advertised server address.
- `TerminalTeleop` still reads stdin from the launching terminal; WebRTC browser keyboard focus does not drive WASD for this viewer.
