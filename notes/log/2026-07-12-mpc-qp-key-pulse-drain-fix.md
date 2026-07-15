# 2026-07-12 MPC QP Key Pulse Drain Fix

## Purpose

Fix the viewer behavior where a short `mpc_qp` key pulse planned one moving trajectory but the next zero-command loop immediately replaced it with final-frame hold.

## Stage

MPC-QP viewer / terminal teleop / direct kinematic playback.

## Related Todo

- [T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Code Change

- Updated `Go2Pvcnn/extension/viz/go2_foostep_planner.py` so `_viewer_should_drain_before_zero_replan()` applies to both `mpc` and `mpc_qp`.
- Added `drain_current_trajectory` to `_viewer_should_hold_mpc_qp_zero_command()` so zero-command final-frame hold yields while the just-planned moving trajectory is still being drained.
- Changed the viewer loop to compute drain-before-zero using the selected backend instead of hard-coded `backend="mpc"`.
- Added a regression test in `Go2Pvcnn/tests/test_mpc_qp_backend.py` for "zero after nonzero should drain active `mpc_qp` motion before holding".

## Verification

Focused unit tests:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q -k "zero_command_drains_active_motion or zero_command_forces_hold or zero_command_hold_uses_previous"
```

Result: `3 passed, 69 deselected`.

Viewer-related tests:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q -k "replan or teleop or step_mode or mpc_planning"
```

Result: `7 passed, 27 deselected`.

Syntax/diff check:

```bash
python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_qp_backend.py
git diff --check -- Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_qp_backend.py
```

Result: both exit `0`.

Real IsaacSim scripted single-pulse test on GPU1:

```bash
timeout -s INT -k 20s 150s env CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --qp-iterations 3 --warmup-steps 0 --scripted-command "0.45 0.0 0.0" --scripted-command-cycles 1 --timing-debug --timing-sync-cuda
```

Result: process reached viewer loop and was stopped by timeout after collecting timing rows.

Key metrics parsed from `/tmp/mpc_qp_scripted_drain_after_patch.log`:

- First nonzero row: cycle `0`, frame `0`, `command_vx=0.45`, `need_replan=true`, `force_zero_hold=false`, `plan_ms=462.737`, `qp_total_ms=456.719`.
- After scripted command returned to zero:
  - frames `1-20` kept `command_vx=0.0`, `need_replan=false`, `force_zero_hold=false`, `plan_ms=None`.
  - playback continued at roughly `20-29ms` per frame instead of immediately replacing the result with final-frame hold.

## Residual Risk

The full `test_mpc_qp_backend.py` suite still has existing unrelated QP planner-side failures in gait/continuous solver/manager API tests. They are not caused by this viewer state-machine change, but remain open T302v work.

## Conclusion

The short-key pulse failure mode is fixed for the viewer path: a single nonzero command can now drain the planned `mpc_qp` motion trajectory after the command returns to zero, instead of being truncated by immediate zero-command hold.

## Git Refs

- Baseline Ref: `8168b15` plus dirty T302v workspace
- Candidate Ref: current dirty workspace
- Key Files:
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
