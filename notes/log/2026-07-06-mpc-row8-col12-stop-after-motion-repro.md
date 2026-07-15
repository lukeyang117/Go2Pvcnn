# MPC Row8 Col12 Stop-After-Motion Repro

## Purpose

Reproduce the user-observed existing `mpc` issue where row `8`, col `12` looks acceptable during a single moving plan but develops large touchdown / actual-foot mismatch after giving velocity and then releasing the command to zero.

## Stage

- Existing `planner_backend="mpc"` viewer/runtime diagnostic.
- This is not `mpc_qp`.

## Related Todo

- [../todo/T302w-mpc-row8-col12-loss-tuning.md](../todo/T302w-mpc-row8-col12-loss-tuning.md)

## Commands

Required terrain:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 12 --requested-n-frames 25 --warmup-steps 4 --playback-frames 25 --sequence 'move_v050:0.50,0.00,0.00x2;stop:0.00,0.00,0.00x4'
```

Control terrain:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 0 --terrain-col 0 --requested-n-frames 25 --warmup-steps 4 --playback-frames 25 --sequence 'move_v050:0.50,0.00,0.00x2;stop:0.00,0.00,0.00x4'
```

Static check:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py
```

## Input Conditions

- No reset between sequence phases.
- Sequence: two moving replans with `vx=0.50`, then four stop replans with zero command.
- Horizon: `25`.
- Playback frames per segment: `25`.

## Key Metrics

Row `8`, col `12`:

- Moving phase max `touchdown_to_current_actual_foot_error_max_m`: `0.34057`
- Moving phase max `touchdown_to_contact_frame_foot_error_max_m`: `0.40622`
- Stop phase stable `planned_vs_fk_foot_error_all_max_m`: `0.17768`
- Stop phase stable `playback_readback_error_max_m`: `0.17768`
- Stop phase stable `touchdown_to_current_actual_foot_error_max_m`: `0.17768`
- Stop phase `raw_ik_joint_limit_violation_max`: `0.83780`
- Stop phase `calf_upper_saturation_max`: `0.83780`

Control row `0`, col `0`:

- Stop phase stable `planned_vs_fk_foot_error_all_max_m`: `0.02500`
- Stop phase stable `playback_readback_error_max_m`: `0.02500`
- Stop phase stable `touchdown_to_current_actual_foot_error_max_m`: `0.02500`
- Stop phase `raw_ik_joint_limit_violation_max`: `0.0`
- Stop phase `calf_upper_saturation_max`: `0.0`

## Result

Reproduced. The static one-shot row `8`, col `12` moving plan still has micron-scale planned-vs-FK error, but the user-described dynamic sequence exposes a real stop-after-motion error:

- row `8`, col `12` stop-phase readback/FK mismatch is about `0.1777m`;
- row `0`, col `0` control stop-phase mismatch is about `0.0250m`;
- the bad row also shows large raw IK/calf-limit violation during stop, which the control row does not.

## Conclusion

The earlier static baseline was insufficient. The actionable reproduction is a continuous sequence: move, then zero command without reset. On row `8`, col `12`, the stop plan asks for terrain-grounded feet that the clamped IK / playback cannot realize, creating the visible foot/touchdown mismatch.

## Follow-Up

- Investigate stop/zero-command branch after motion on sloped/high terrain.
- Compare standstill fallback or zero-command parametric export against actual current FK feet before changing production loss.
- Any fix must keep the 2026-05-28 low-small acceptance metrics intact and must not touch `mpc_qp`.

## Git Refs

- Baseline Ref: `8168b15`
- Candidate Ref: `8168b15` plus dirty working tree on 2026-07-06
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py](../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
