# 2026-07-11 MPC QP Viewer Key Latency Timing

## Purpose

Measure where the `mpc_qp` viewer spends time after a nonzero velocity command reaches the viewer loop.

## Stage

MPC-QP viewer / direct kinematic playback timing.

## Related Todo

- [T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Command

```bash
timeout -s INT -k 20s 140s env CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --qp-iterations 3 --warmup-steps 0 --scripted-command "0.45 0.0 0.0" --scripted-command-cycles 1 --timing-debug --timing-sync-cuda
```

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- GPU: `CUDA_VISIBLE_DEVICES=1`, selected as `cuda:0`
- Viewer backend: `mpc_qp`
- Horizon: `25`
- `qp_iterations`: `3`
- Timing mode: `--timing-debug --timing-sync-cuda`

## Key Metrics

Startup:

- Terrain generation: `2.125017s`
- Scene creation: `31.226985s`
- Simulation start: `12.448397s`
- Headless stdin warning: `stdin is not a TTY; teleop keys are disabled`

First nonzero command timing row:

- `loop_until_playback_ms`: `546.455`
- `plan_ms`: `484.732`
- `qp_total_ms`: `478.161`
- `qp_nominal_ms`: `208.392`
- `qp_solve_ms`: `257.259`
- `qp_diagnostics_ms`: `11.188`
- `state_read_ms`: `10.328`
- `terrain_build_ms`: `2.491`
- `semantic_viz_sample_ms`: `1.016`
- `visualizer_update_ms`: `11.465`
- `playback_ms`: `32.433`
- `teleop_poll_ms`: `0.039`

After scripted command returned to zero:

- Rows: `130`
- Mean `loop_until_playback_ms`: `22.240`
- Mean `plan_ms`: `0.261`
- Mean `playback_ms`: `12.694`

## Result

The first nonzero command reached the planner and did not crash. The slow part was the first `mpc_qp` plan itself, not keyboard polling or terrain extraction. Within the plan, `qp_iterations=3` spent most time in nominal warm start plus continuous QP solve.

The full viewer startup/reload cost was much larger than one QP prediction: scene creation and simulation start alone were about `43.7s`, before the viewer loop.

## Conclusion

For the user's perceived delay:

- If the viewer is already loaded and the command really reaches the loop, the first `mpc_qp` prediction with `qp_iterations=3` costs about `0.48s` on this run.
- `teleop_poll_ms` was tiny in the scripted run, but headless WebRTC printed `stdin is not a TTY; teleop keys are disabled`, so keyboard input through the livestream window is not guaranteed to reach `TerminalTeleop`.
- The current zero-command hold path can repeatedly emit very light hold/replan rows after scripted command release; this is not the first nonzero-command bottleneck but should be tracked separately if it affects interactive smoothness.

## Follow-Up

- Use the same command with `--timing-debug --timing-sync-cuda` on an interactive terminal run to distinguish "key did not enter stdin" from "key entered and QP took ~0.5s".
- If speed must improve without changing trajectory behavior, first target the `qp_nominal_ms + qp_solve_ms` path for `qp_iterations=3`.

## Git Refs

- Baseline Ref: `8168b15` plus dirty T302v workspace
- Candidate Ref: current dirty workspace
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
