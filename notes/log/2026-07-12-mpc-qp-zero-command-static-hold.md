# 2026-07-12 MPC QP Zero-Command Static Hold

## Purpose

Implement and verify the viewer policy requested by the user: for `mpc_qp`, zero velocity should not enter QP planning when idle; only nonzero velocity should run the QP optimized trajectory. If a nonzero trajectory already exists, zero command should still drain that trajectory before returning to static hold.

## Stage

MPC-QP viewer / direct kinematic playback / input latency.

## Related Todo

- [T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Code Change

- Added `_viewer_should_static_hold_mpc_qp_zero_command()` in `Go2Pvcnn/extension/viz/go2_foostep_planner.py`.
- Added `_viewer_static_hold_result_from_current_state()` to build a stationary `ViewerTrajectoryResult` from the current robot state without terrain construction or QP planning.
- Moved the `mpc_qp` zero-command hold branch before terrain build in the viewer replan path.
- Preserved the existing drain behavior: after a nonzero command plans a moving trajectory, the following zero command keeps playing that trajectory until it lands/drains.
- Did not change the existing `mpc` backend or QP losses.

## Verification

RED test:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q -k "zero_command_without_result_uses_static_hold_path or static_hold_result_from_current_state"
```

Result before implementation: `2 failed` with missing helper attributes.

GREEN focused tests:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q -k "zero_command_without_result_uses_static_hold_path or static_hold_result_from_current_state or zero_command_drains_active_motion or zero_command_forces_hold or zero_command_hold_uses_previous"
```

Result: `5 passed, 69 deselected`.

Viewer regression:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q -k "replan or teleop or step_mode or mpc_planning"
```

Result: `7 passed, 27 deselected`.

Syntax/diff:

```bash
python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_qp_backend.py
git diff --check -- Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_qp_backend.py
```

Result: both exit `0`.

Real idle viewer test on card 2:

```bash
timeout -s INT -k 20s 150s env CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --qp-iterations 3 --timing-debug --timing-sync-cuda --idle-debug --idle-debug-stride 1
```

Log: `/tmp/mpc_qp_zero_static_hold_after_patch.log`.

Key metrics:

- Timing rows: `27`
- Zero-command timing rows: `27`
- Zero-command rows with QP fields: `0`
- Zero-command rows with `terrain_build_ms`: `0`
- Max zero-command `plan_ms`: `0.791ms`
- Max idle root delta: `0.0m`
- Max idle foot delta: `0.0m`

Real scripted nonzero pulse on card 2:

```bash
timeout -s INT -k 20s 150s env CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --qp-iterations 3 --warmup-steps 0 --scripted-command "0.45 0.0 0.0" --scripted-command-cycles 1 --timing-debug --timing-sync-cuda --idle-debug --idle-debug-stride 1
```

Log: `/tmp/mpc_qp_nonzero_pulse_after_zero_static_patch.log`.

Key metrics:

- Nonzero timing rows: `1`
- Nonzero first plan: `plan_ms=966.18`, `qp_total_ms=947.51`, `qp_solve_ms=592.89`
- Post-nonzero zero frames in cycle `0` had `need_replan=false`, confirming trajectory drain.
- After drain, zero-command static hold rows had `plan_ms < 1ms` and no terrain build.

## Result

Pass. Idle zero command no longer calls full `mpc_qp` or builds terrain in the viewer replan path. Nonzero command still runs QP, and release-to-zero still drains the generated trajectory before static hold.

## Follow-Up

- Remaining real user latency may still include terminal-vs-WebRTC input routing, because current keyboard control is still `sys.stdin` based.
- Nonzero QP on the loaded card 2 run was close to `1s`; this is expected to remain until QP/nominal compute is optimized.

## Git Refs

- Baseline Ref: `8168b15` plus dirty T302v workspace
- Candidate Ref: current dirty workspace
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
