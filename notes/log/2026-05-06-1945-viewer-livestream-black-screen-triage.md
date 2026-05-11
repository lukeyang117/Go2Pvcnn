# Viewer Livestream Black Screen Triage

## Purpose

Triage the user report that `Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --device cuda:0 --num_envs 1 --terrain task --planner-backend together` starts with WebRTC/NvStreamer errors, the visualizer shows no image, and the run later appears to stall after GLFW/windowing warnings.

## Stage

`extension/viz` remote WebRTC viewer startup and terminal teleop input, under T100/T109/T110 follow-up.

## Related Todo

[T100/T111](../todo/T100-batched-together-planner-gpu-migration.md#t111-viewer-livestream-black-screen--glfw-disconnect-triage)

## Command / Procedure

- Read repository entrypoint notes and planner/viewer notes:
  - [../index.md](../index.md)
  - [../todo.md](../todo.md)
  - [../human/human-08-extension-planner-reading-guide.md](../human/human-08-extension-planner-reading-guide.md)
  - [../human/human-09-extension-planner-mapping.md](../human/human-09-extension-planner-mapping.md)
  - [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)
  - [../human/human-14-batched-planner-viewer-diagnostics-summary.md](../human/human-14-batched-planner-viewer-diagnostics-summary.md)
- Read viewer implementation:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
- Checked local process and GPU state:

```bash
ps -eo pid,ppid,stat,etime,cmd | rg -i "go2_foostep|isaac|omni|kit|nvstream|livestream|webrtc"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
nvidia-smi
ps -p 581038 -o pid,ppid,stat,etime,cmd
```

## Input Conditions

- User-reported command used `--headless --livestream 2` with `planner-backend together`.
- User log included repeated `carb.livestream-rtc.plugin nvstPushStreamData error ... 0x800b0000`, `NVST_CCE_DISCONNECTED`, connection-count underflow-looking values, `GLFW initialization failed`, and `Failed to startup plugin carb.windowing-glfw.plugin`.
- Current workspace already contains generated `${data}/NvStreamer-20260506-*.etli` files from recent livestream attempts.

## Key Metrics

- Code contract:
  - `_prepare_runtime_args()` enables `--enable_cameras` automatically when `livestream in (1, 2)`.
  - `gym.make(..., render_mode="rgb_array")` is used only for livestream mode.
  - `TerminalTeleop` reads keyboard input from process `stdin`, not from the WebRTC browser/window.
  - The current playback branch already calls `_viewer_direct_playback_step()` with scene sync: `robot write -> scene.write_data_to_sim() -> sim.render() -> scene.update(physics_dt)`.
- Process check:
  - No obvious `go2_foostep_planner`, Isaac, Kit, NvStreamer, or WebRTC process remained after the report.
  - GPU had an unrelated long-running `python continual_learning/train_continual.py --task go2_amp ... --sim_device cuda:0 --wm_device cuda:1 ...` process, PID `581038`.
  - That process used about `12436 MiB` on GPU 0 and `19150 MiB` on GPU 1.

## Result

Diagnostic partial.

The observed log pattern is more consistent with a WebRTC/livestream client or rendering-backend connection problem than with the together planner core. The `omni.usd-abi.plugin` missing `/rtx-defaults/...` setting warnings are likely non-fatal noise. The repeated `nvstPushStreamData` errors and `NVST_CCE_DISCONNECTED` messages are the stronger signal for no remote image. `GLFW initialization failed` is expected to be risky in headless contexts and can appear when a local windowing backend is attempted or initialized during a headless/livestream run.

## Conclusion

For this report, the first split should be:

1. Run a scripted headless smoke with `--livestream 0` to prove Isaac/env/planner/playback reaches `[Viewer][Plan]` and `[Viewer][Playback]`.
2. Run the same with `--livestream 2` to isolate WebRTC/RTC streaming.
3. Treat browser/window `WASD` as unrelated to teleop until stdin is confirmed, because viewer teleop is terminal-driven.

## Follow-Up

- Prefer using a scripted command during black-screen triage:

```bash
/home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 0 \
  --device cuda:2 \
  --num_envs 1 \
  --terrain task \
  --planner-backend together \
  --warmup-steps 0 \
  --scripted-command "0.20 0.00 0.00" \
  --scripted-command-cycles 1
```

- If `--livestream 0` reaches `[Viewer][Plan]` and `[Viewer][Playback]`, retry with `--livestream 2`; any black screen is then a streaming/browser/client issue, not planner startup.
- For interactive teleop over WebRTC, keep focus on the launching terminal or add a future viewer-keyboard event path; the current viewer does not read WebRTC window key events.

## Git Refs

- Baseline Ref: `working tree on 2026-05-06`
- Candidate Ref: no code changes
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
