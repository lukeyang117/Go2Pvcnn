# 2026-05-06 20:54 IsaacLab Livestream Extension Dedupe Fix

## Scope

- Todo: [T100/T111 viewer livestream black-screen / GLFW disconnect triage](../todo/T100-batched-together-planner-gpu-migration.md#t111-viewer-livestream-black-screen--glfw-disconnect-triage)
- Stage: IsaacLab `AppLauncher` + `Go2Pvcnn/extension/viz` WebRTC viewer startup
- Environment: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`

## Context

The first remote WebRTC fix made `Go2Pvcnn/extension/viz/go2_foostep_planner.py` advertise a non-loopback `PUBLIC_IP`, but the user still reported a popped local visualization window with black output and server-side errors:

- `carb.livestream-rtc.plugin nvstPushStreamData timeout for eye 0`
- `carb.livestream-rtc.plugin nvstPushStreamData error ... 0x800b0000`
- `main: thread_init: already added for thread`

## Root Cause

IsaacLab `AppLauncher` in [../../../IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py](../../../IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py) appended livestream arguments in two phases for `--livestream 2`:

1. `_resolve_livestream_settings()` appended `--enable omni.kit.livestream.webrtc`.
2. `_resolve_experience_file()` later appended `--enable omni.services.livestream.nvcf` plus `publicEndpointAddress` and `port`.

A dry-run before the fix showed both livestream implementations in `sys.argv`, despite the source comment saying only one livestream extension can be enabled at a time. This matched the client-connect-time stream push failures better than the non-fatal `GLFW initialization failed` warning.

## Change

- Made `_resolve_livestream_settings()` side-effect free for Kit argv.
- Kept livestream mode validation and environment precedence there.
- Left actual livestream Kit argument construction to `_resolve_experience_file()`, where the experience file and WebRTC `PUBLIC_IP` are already known.

After the change, `--livestream 2` adds only:

```text
--/app/livestream/publicEndpointAddress=<PUBLIC_IP>
--/app/livestream/port=49100
--enable
omni.services.livestream.nvcf
```

## Verification

- Dry-run after the fix:
  - `livestream=2` added `4` args.
  - No `omni.kit.livestream.webrtc` remained in the added args.
  - `livestream=1` still resolved to the native livestream extension set.
- Compile/diff checks:
  - `python -m py_compile` passed for IsaacLab `app_launcher.py`, viewer script, and viewer tests.
  - `git diff --check` passed for IsaacLab and project files.
- Real server-side smoke:
  - Command used `env_isaacsim`, `--livestream 2`, `--webrtc-public-ip 172.31.179.75`, `--device cuda:2`, `--terrain task`, `--planner-backend together`.
  - Log reached `Streaming server started.`, `Completed setting up the environment...`, `[Viewer][Plan]`, and `[Viewer][Playback]`.
  - Log scan found no `nvstPushStreamData`, `timeout for eye`, or `error for eye`.
  - Process exited by test timeout/INT with exit code `0`; no residual viewer/Isaac GPU process remained.

## Caveats

- Browser-side visual confirmation from the user's local machine is still pending.
- If the local client still black-screens, the next layer is client/WebRTC/network rather than the server-side AppLauncher double-extension bug.
- `GLFW initialization failed` still appears in headless livestream startup, but the server-side scripted viewer path reached render/playback after that warning.
