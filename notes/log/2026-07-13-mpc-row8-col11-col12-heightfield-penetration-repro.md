# 2026-07-13 MPC Row8 Col11/Col12 Heightfield Penetration Repro

## Purpose

- Quantify the user-observed `planner_backend=mpc` issue on `terrain-row=8` with `terrain-col=11` and `terrain-col=12`.
- Focus on foot penetration, body/leg penetration, root motion, touchdown/contact mismatch, and front-foot crossing reluctance.

## Stage

- Existing MPC diagnostics / viewer-runtime fixture.

## Related Todo

- [T302w MPC Row8 Col12 Loss Tuning](../todo/T302w-mpc-row8-col12-loss-tuning.md)

## Command / Procedure

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 11 --requested-n-frames 25 --playback-frames 25 --warmup-steps 4 --commands 'forward_v050:0.50,0.00,0.00;diag_v050:0.35,0.20,0.00' --cycles 1
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 12 --requested-n-frames 25 --playback-frames 25 --warmup-steps 4 --commands 'forward_v050:0.50,0.00,0.00;diag_v050:0.35,0.20,0.00' --cycles 1
```

Before these successful runs, `CUDA_VISIBLE_DEVICES=2` failed during IsaacLab initialization because the mapped GPU had only about `83.94 MiB` free and the semantic raycaster OOMed.

## Input Conditions

- Backend: `mpc`, not `mpc_qp`.
- `requested_n_frames=25`, `playback_frames=25`, `warmup_steps=4`.
- Commands:
  - `forward_v050: 0.50, 0.00, 0.00`
  - `diag_v050: 0.35, 0.20, 0.00`
- Probe-only diagnostics were extended to report heightfield clearances for planned/FK foot, FK knee/shank, and body footprint samples.

## Key Metrics

| Tile | Key Result |
| --- | --- |
| row8/col11 | `max_terminal_planned_vs_fk_foot_error_m=0.023919`; `min_fk_foot_ground_clearance_m=-0.004627`; `max_fk_foot_ground_penetration_count=2`; knee/shank/body penetration counts all `0`; `max_root_step_z_m=0.001490`; `max_touchdown_to_contact_frame_foot_error_m=0.262022`. |
| row8/col12 | `max_terminal_planned_vs_fk_foot_error_m=0.233233`; `min_fk_foot_ground_clearance_m=-0.064212`; `min_fk_knee_ground_clearance_m=-0.048161`; `min_fk_shank_ground_clearance_m=-0.049671`; `min_body_ground_clearance_m=-0.042370`; max penetration counts: foot `23`, knee `11`, shank `28`, body `28`; `max_root_step_z_m=0.120373`; `max_touchdown_to_contact_frame_foot_error_m=0.438814`. |

Per-command row8/col12 details:

- Forward `v=0.50`: FK foot penetration count `17`, min FK foot clearance `-0.044455m`, terminal planned-vs-FK `0.188628m`, calf saturation `0.837800`.
- Diagonal `vx=0.35, vy=0.20`: FK foot/knee/shank/body penetration counts `23/11/28/28`, min clearances `-0.064212/-0.048161/-0.049671/-0.042370m`, terminal planned-vs-FK `0.233233m`, root z step `0.120373m`, calf saturation `0.837800`.

## Result

- Reproduced quantitatively on row8/col12.
- Row8/col12 shows true heightfield penetration in realized FK geometry, not just a visualization issue.
- Row8/col12 also shows IK reachability failure through calf saturation and large planned-vs-FK error.
- Row8/col11 did not reproduce the same body/knee/shank penetration in this two-command probe; it only shows a small foot penetration case near `4.6mm`.

## Conclusion

- The problematic tile is strongly row8/col12 under these commands.
- `crossing_leg_count=0` on both tiles because the existing crossing metric is semantic-obstacle oriented; for these rough terrain tiles, the more relevant evidence is root progress with high touchdown/contact mismatch and FK heightfield penetration.

## Follow-Up

- Add or reuse a hard-terrain-specific crossing/progress metric if the next task needs to prove "root moves but front feet do not cross" independent of semantic obstacles.
- Optimize row8/col12 using the allowed existing MPC loss/candidate-score surfaces without changing metric definitions.

## Git Refs

- Baseline Ref: `8168b15` plus dirty working tree.
- Candidate Ref: `8168b15` plus dirty working tree.
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py](../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py)
  - [../todo/T302w-mpc-row8-col12-loss-tuning.md](../todo/T302w-mpc-row8-col12-loss-tuning.md)
