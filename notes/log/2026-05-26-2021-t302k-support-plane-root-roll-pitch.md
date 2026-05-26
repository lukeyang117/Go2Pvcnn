# 2026-05-26 20:21 T302k Support-Plane Root Roll Pitch

## Purpose

Verify and fix the user-reported downstairs/sloped-terrain issue where parametric MPC left root roll/pitch horizontal instead of correcting the root attitude from the foot support plane.

## Stage

`extension/batch_mpc_planner` parametric decode.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Command / Procedure

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/tests/test_batch_mpc_parametric.py
```

Also ran a small `env_isaacsim` script with front feet at lower terrain height and rear feet at higher terrain height to print decoded root roll/pitch.

## Input Conditions

- Batch size `1`.
- Horizon `25`.
- Root starts with zero roll/pitch.
- Terrain height decreases along `+x`, matching a downhill/downstairs support-plane proxy.
- Front feet are lower than rear feet.

## Key Metrics

- Red test before fix: terminal root pitch stayed `0.0`.
- Fixed focused pytest: `12 passed`.
- Fixed `env_isaacsim` pytest: `12 passed`.
- Fixed pycompile: pass.
- `env_isaacsim` numeric script:
  - `frame0_roll_pitch = [0.0, 0.0]`
  - `terminal_roll_pitch = [0.0, 0.23541666567325592]`
  - `terminal_pitch_rad = 0.23541666567325592`

## Result

Pass. Parametric decode now fits a contact-weighted foot support plane in the root-yaw frame and ramps root roll/pitch toward that estimate after frame0. Frame0 still preserves the current IsaacLab root roll/pitch to avoid replan discontinuity.

## Conclusion

The observed horizontal-root issue was in `decode_parametric_trajectory`: root roll/pitch copied the initial state for the whole horizon while only yaw changed. The fix updates only the parametric decode attitude generation; yaw logic and viewer command-frame behavior remain unchanged.

## Follow-Up

Run a real viewer/IsaacLab stair tile playback if visual confirmation is needed. T302k.12 touchdown-current mismatch and high/large semantic acceptance remain separate open issues.

## Git Refs

- Baseline Ref: working tree on top of `1b799cd`
- Candidate Ref: working tree on top of `1b799cd`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
