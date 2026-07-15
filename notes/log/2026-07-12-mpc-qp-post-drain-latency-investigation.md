# 2026-07-12 MPC QP Post-Drain Latency Investigation

## Purpose

Investigate the user's follow-up report that after the key-pulse drain fix, pressing a key still takes more than one second before visible robot motion.

## Stage

MPC-QP viewer / interactive input / direct kinematic playback timing.

## Related Todo

- [T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

1. Ran a scripted single-pulse viewer test with both `--timing-debug` and `--idle-debug` to separate QP time from actual root/foot motion:

```bash
timeout -s INT -k 20s 150s env CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --qp-iterations 3 --warmup-steps 0 --scripted-command "0.45 0.0 0.0" --scripted-command-cycles 1 --timing-debug --timing-sync-cuda --idle-debug --idle-debug-stride 1
```

2. Ran an interactive TTY viewer with `--timing-debug --idle-debug`, waited until idle loop, then injected one `w` into stdin.

3. Searched the viewer code for keyboard input paths.

## Key Metrics

Scripted single-pulse test (`/tmp/mpc_qp_scripted_idle_motion_after_patch.log`):

- First nonzero row: `plan_ms=484.968`, `qp_total_ms=479.093`, `loop_until_playback_ms=565.294`.
- Frame `1` after command returned to zero: `root_delta_m=0.04036`, `foot_delta_max_m=0.02139`.
- Cumulative root motion exceeded `5cm` by frame `2`, i.e. about `0.04s` of playback after planning.

Interactive TTY injection (`/tmp/mpc_qp_interactive_after_patch_latency.log`):

- Injected `w` was observed as `command_vx=0.5` at cycle `90`.
- Nonzero plan: `plan_ms=274.529`, `qp_total_ms=271.486`, `loop_until_playback_ms=350.081`.
- Next frame after command returned to zero: `root_delta_m=0.08376`, `foot_delta_max_m=0.18718`.
- Frames after the key pulse kept `need_replan=false`, `force_zero_hold=false`, so the previous drain fix remained active.

Idle/zero observations:

- Current viewer code has no Omniverse/WebRTC keyboard subscription; keyboard control is only through `TerminalTeleop`, which reads `sys.stdin`.
- GPU status during investigation showed card `1` idle, while cards `0/2/3` were busy.
- Occasional zero-command full QP plans still occur after a motion cycle, measured here at `206-360ms`. This is not a ten-second delay by itself, but it can add perceptible stalls under GPU contention.

## Result

The "trajectory is computed but takes more than one second to visibly move" path did not reproduce on GPU1:

- Once a command entered the viewer loop, visible root motion appeared on the next playback frame after QP planning.
- The measured command-to-first-playback time was `~0.35s` in interactive TTY and `~0.56s` in scripted headless.

The likely remaining issue is at the input boundary or GPU scheduling boundary:

- If the user presses keys in the WebRTC/browser window, those events are not wired into the current `TerminalTeleop` path. They may not reach `sys.stdin` at all.
- If the selected CUDA device is busy, the first QP plan can be delayed beyond the GPU1 numbers.

## Follow-Up

- To confirm the user's exact run, launch with `--timing-debug --timing-sync-cuda --idle-debug --idle-debug-stride 1` and inspect whether a pressed key produces a `viewer_timing_debug` row with nonzero `command_vx`.
- If browser-window control is required, add an explicit Omniverse/WebRTC keyboard input path instead of relying only on terminal stdin.
- If terminal stdin is confirmed and latency is still high, profile the first `qp_total_ms` on the user's chosen GPU and optimize the `qp_nominal_ms + qp_solve_ms` path.

## Git Refs

- Baseline Ref: `8168b15` plus dirty T302v workspace
- Candidate Ref: current dirty workspace
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
