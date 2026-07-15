# MPC Row8 Col12 Nonzero Weight Tuning

## Purpose

Quantify the user's latest row `8`, col `12` existing-`mpc` complaint under nonzero velocity commands and tune only existing MPC loss weights. This pass explicitly does not target the zero-command stop branch as the primary acceptance path.

## Stage

- Existing `planner_backend="mpc"` parametric planner.
- Isolated from `mpc_qp`; no `Go2Pvcnn/extension/batch_mpc_qp_planner/` file was edited.

## Related Todo

- [../todo/T302w-mpc-row8-col12-loss-tuning.md](../todo/T302w-mpc-row8-col12-loss-tuning.md)

## Commands

Baseline nonzero matrix:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 12 --requested-n-frames 25 --warmup-steps 4 --playback-frames 25 --commands 'fwd_v030:0.30,0.00,0.00;fwd_v050:0.50,0.00,0.00;fwd_v070:0.70,0.00,0.00;back_v030:-0.30,0.00,0.00;lat_l_v040:0.00,0.40,0.00;lat_r_v040:0.00,-0.40,0.00;diag_l_v050:0.35,0.35,0.00;diag_r_v050:0.35,-0.35,0.00;mixed_l:0.50,0.25,1.00;mixed_r:0.50,-0.25,-1.00'
```

Candidate/final default matrix after weight update:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 12 --requested-n-frames 25 --warmup-steps 4 --playback-frames 25 --commands 'fwd_v030:0.30,0.00,0.00;fwd_v050:0.50,0.00,0.00;fwd_v070:0.70,0.00,0.00;back_v030:-0.30,0.00,0.00;lat_l_v040:0.00,0.40,0.00;lat_r_v040:0.00,-0.40,0.00;diag_l_v050:0.35,0.35,0.00;diag_r_v050:0.35,-0.35,0.00;mixed_l:0.50,0.25,1.00;mixed_r:0.50,-0.25,-1.00'
```

2026-05-28 low-small acceptance:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants baseline --cycles 1 --requested-n-frames 25 --warmup-steps 6
```

Focused static tests:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q -k 'touchdown_endpoint_loss_uses_config_weight or root_foot_center_loss_uses_config_weight or parametric_losses_include_endpoint_and_foot_height_guards or root_rpy'
```

## Input Conditions

- Backend: `planner_backend="mpc"`.
- Terrain: row `8`, col `12`.
- Horizon: `25`.
- Plan dt: `0.02`.
- Nonzero command matrix covers forward speeds, backward, lateral left/right, diagonal left/right, and mixed translation/yaw.

## Key Metrics

Baseline nonzero matrix:

- `max_playback_readback_error_m = 4.5942e-6`
- `max_terminal_planned_vs_fk_foot_error_m = 2.7079e-6`
- `max_raw_ik_joint_limit_violation ~= 1.3e-5`
- `max_touchdown_to_contact_frame_foot_error_m = 0.53763`
- `max_touchdown_to_current_actual_foot_error_m = 0.61080`
- `max_fk_semantic_collision_count = 0`

Final default matrix after weight update:

- `ik_fk_residual.weight = 16.0`
- `kinematics.weight = 3.0`
- `root_foot_center.weight = 4.0`
- `touchdown_endpoint.weight = 16.0`
- `progress.weight = 0.35`
- `swing_direction.weight = 0.25`
- `max_playback_readback_error_m = 5.4001e-6`
- `max_terminal_planned_vs_fk_foot_error_m = 4.2716e-6`
- `max_raw_ik_joint_limit_violation = 2.3365e-5`
- `max_touchdown_to_contact_frame_foot_error_m = 0.47605`
- `max_touchdown_to_current_actual_foot_error_m = 0.60490`
- `max_fk_semantic_collision_count = 0`

2026-05-28 low-small hard metrics after weight update:

- diagonal `crossing_leg_count = 1`
- diagonal `fk_semantic_collision_count = 0`
- diagonal `fk_semantic_min_clearance_over_semantic_m = 0.08004`
- diagonal `planned_vs_fk_foot_error_crossing_leg_max_m = 9.7759e-7`
- summary `max_terminal_planned_vs_fk_foot_error = 9.7759e-7`
- probe exit code `0`

Focused tests:

- `4 passed, 207 deselected`
- pycompile exit code `0`

## Result

Partial improvement for the visible touchdown-marker path while preserving FK/IK:

- Nonzero velocity planned-vs-FK and playback readback were already micron-scale at baseline and remain micron-scale after tuning.
- Raw IK limit violation remains tiny, increasing from roughly `1e-5rad` to `2.34e-5rad`, still not a material IK failure.
- The direct contact-frame touchdown metric improved from `0.53763m` to `0.47605m`.
- The current-foot touchdown metric remains high because `planned_touchdown_w` is a future touchdown marker and can legitimately be far from the current foot at frame 0; this metric should not be treated as planned-vs-FK.
- Low-small spec hard metrics were not broken.

## Conclusion

The user's nonzero-speed matrix does not show a true FK/IK mismatch: FK/readback is micron-scale across speed and direction. The large visible discrepancy is concentrated in touchdown-marker alignment semantics. The best tested weight-only update strengthens existing endpoint/FK/IK/root-foot terms and reduces progress/swing authority, improving contact-frame touchdown alignment without adding or deleting losses.

## Follow-Up

- If the user wants current-foot-to-touchdown marker distance to be a hard gate, the marker semantics may need to change from "future touchdown" to "nearest/current realized touchdown"; that is not a loss-weight-only change.
- The stop-after-motion branch remains separately reproduced in [2026-07-06-mpc-row8-col12-stop-after-motion-repro.md](2026-07-06-mpc-row8-col12-stop-after-motion-repro.md).

## Git Refs

- Baseline Ref: `8168b15`
- Candidate Ref: `8168b15` plus dirty working tree on 2026-07-06
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py](../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
