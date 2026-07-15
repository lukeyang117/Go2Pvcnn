# 2026-07-12 MPC QP Real-Chain Input A/B

## Purpose

Mimic the user's real viewer command chain and quantify whether keyboard input reaches the `mpc_qp` viewer through the current implementation.

## Stage

MPC-QP viewer / headless livestream / input boundary diagnostics.

## Related Todo

- [T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

Both runs use the user's real viewer shape, with timing diagnostics appended:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:0 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc_qp \
  --n-frames 25 \
  --plan-dt 0.02 \
  --qp-iterations 3 \
  --timing-debug \
  --timing-sync-cuda \
  --idle-debug \
  --idle-debug-stride 1
```

Run A: no TTY, WebRTC/headless process only. Log: `/tmp/mpc_qp_real_chain_notty_stdin_w.log`.

Run B: TTY allocated, then one `w` injected into terminal stdin after viewer loop started. Log: `/tmp/mpc_qp_real_chain_tty_stdin_w.log`.

## Metrics

Run A, no TTY:

- `stdin is not a TTY`: `true`
- timing rows: `32`
- idle rows: `795`
- nonzero timing rows: `0`
- attempted stdin write from the test harness failed because stdin was closed for the non-TTY session
- max zero-command expensive plan: `252.064ms`

Run B, TTY stdin:

- `stdin is not a TTY`: `false`
- timing rows: `53`
- idle rows: `984`
- nonzero timing rows: `1`
- nonzero row:
  - cycle `20`, frame `0`
  - `command_vx=0.5`
  - `loop_until_playback_ms=392.818`
  - `plan_ms=305.991`
  - `qp_total_ms=301.922`
  - `teleop_poll_ms=0.164`
  - `playback_ms=68.547`
- first motion after nonzero:
  - cycle `20`, frame `1`
  - `root_delta_m=0.08149`
  - `foot_delta_max_m=0.02953`
- max zero-command expensive plan: `260.614ms`

## Result

The A/B result supports the input-boundary hypothesis:

- With the same headless/livestream viewer chain but no TTY, the current viewer disables `TerminalTeleop` and no nonzero command is observed.
- With TTY stdin, a single `w` reaches the planner and visible root motion appears on the next playback frame after a `~0.39s` loop.

This means current keyboard control is not a WebRTC/browser-window input path. It is a terminal-stdin input path. If the user presses keys inside the livestream/browser window, current code has no measured path that forwards those events into `active_cmd.values`.

## Conclusion

The remaining user-observed "press key and wait / sometimes no movement" behavior is not explained by playback drain after command entry. It is explained by the missing browser/WebRTC keyboard input path, plus secondary GPU contention/QP timing effects.

## Follow-Up

- Add a real Omniverse/WebRTC keyboard input bridge, or expose a deterministic network/control-channel input path for the viewer.
- Keep `TerminalTeleop` as a fallback for local terminal operation.
- Preserve current `mpc_qp` drain behavior once any input source produces a nonzero command.

## Git Refs

- Baseline Ref: `8168b15` plus dirty T302v workspace
- Candidate Ref: current dirty workspace
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
