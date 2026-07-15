# 2026-07-12 MPC QP Interactive Key Hold Repro

## Purpose

Reproduce the user's report that pressing viewer keys can feel much slower than the measured sub-second `mpc_qp` planning time, and some presses appear to do nothing.

## Stage

MPC-QP viewer / interactive terminal teleop / direct kinematic playback.

## Related Todo

- [T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Command

```bash
env CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless --livestream 2 --webrtc-public-ip 172.31.179.75 \
  --device cuda:0 --num_envs 1 --terrain task \
  --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 \
  --qp-iterations 3 --warmup-steps 0 \
  --timing-debug --timing-sync-cuda \
  2>&1 | tee /tmp/mpc_qp_interactive_stdin_timing_card1.log
```

Then injected `w`, repeated `w`, and Ctrl-C through the attached TTY session.

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- GPU: `CUDA_VISIBLE_DEVICES=1`, selected as `cuda:0`
- Viewer backend: `mpc_qp`
- Horizon: `25`
- `qp_iterations`: `3`
- Default `--key-hold-timeout`: `0.18s`
- Timing mode: `--timing-debug --timing-sync-cuda`

## Key Metrics

Parsed from `/tmp/mpc_qp_interactive_stdin_timing_card1.log`:

- Timing rows: `378` before the final extended run sample.
- Nonzero command rows: `2`, at cycles `322` and `340`.
- Nonzero rows:
  - Cycle `322`: `command_vx=0.5`, `plan_ms=289.681`, `qp_total_ms=286.226`, `loop_until_playback_ms=366.857`.
  - Cycle `340`: `command_vx=0.5`, `plan_ms=263.866`, `qp_total_ms=260.230`, `loop_until_playback_ms=343.141`.
- Immediately after each nonzero row:
  - Cycle `323`: `command_vx=0`, `force_zero_hold=true`, `plan_ms=None`, `playback_ms=23.068`.
  - Cycle `341`: `command_vx=0`, `force_zero_hold=true`, `plan_ms=None`, `playback_ms=23.132`.
- Zero-command `need_replan=true` rows: `376`.
- Zero-command loop mean: `23.086ms`; `teleop_poll_ms` mean: `0.052ms`.

## Result

The key input can enter `TerminalTeleop`; it is not purely a missing-stdin problem in this TTY run. However, each `w` pulse produced only one nonzero command row. The following loop saw a zero command and the `mpc_qp` zero-command path immediately replaced the motion trajectory with a final-frame hold.

Because the default key hold timeout is `0.18s` while the first nonzero `qp_iterations=3` plan took about `0.26-0.29s`, the key expires before the next loop can keep playing the planned trajectory. This explains why a short key press can look like no movement or a tiny delayed movement, even though the QP solve itself is well under one second in this run.

## Conclusion

The current symptom is a viewer interaction/state-machine issue:

- Planning is still a real cost, about `0.26-0.29s` for the measured nonzero cycles.
- The larger perceived delay comes from short key events being converted back to zero before playback can advance beyond the first frame.
- The `mpc_qp` zero-command hold path then truncates the just-planned moving trajectory.
- WebRTC/browser focus may still add another layer, but the TTY test alone reproduces the core failure mode.

## Follow-Up

No code was changed in this pass. A fix should preserve the previous user contract: after a nonzero key pulse, play/drain the generated trajectory like `mpc` instead of instantly zero-holding it, while keeping the idle zero-command anti-jitter behavior for true idle.

## Git Refs

- Baseline Ref: `8168b15` plus dirty T302v workspace
- Candidate Ref: current dirty workspace
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
