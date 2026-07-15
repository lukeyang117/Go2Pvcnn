# 2026-07-13 MPC Row8 Col11/Col12 Body Shank Clearance Fix

## Purpose

- Verify the existing `planner_backend=mpc` rough-terrain fix after the farther-walk row `8`, col `11` / col `12` penetration repro.
- Preserve the old low-small redesign hard metrics from `docs/superpowers/specs/2026-05-28-parametric-low-small-loss-redesign.html`.

## Stage

- Existing MPC parametric planner / FK body-leg collision safety.

## Related Todo

- [T302w MPC Row8 Col12 Loss Tuning](../todo/T302w-mpc-row8-col12-loss-tuning.md)

## Command / Procedure

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 11 --requested-n-frames 25 --playback-frames 25 --warmup-steps 4 --commands 'forward_v050:0.50,0.00,0.00' --cycles 6
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 12 --requested-n-frames 25 --playback-frames 25 --warmup-steps 4 --commands 'forward_v050:0.50,0.00,0.00' --cycles 6
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants baseline --cycles 1 --requested-n-frames 25 --warmup-steps 6
```

## Input Conditions

- Backend: `mpc`, not `mpc_qp`.
- Visible GPU: `CUDA_VISIBLE_DEVICES=1`, script device `cuda:0`.
- `requested_n_frames=25`, `playback_frames=25`, `warmup_steps=4`.
- Runtime optimizer steps remained `24`.

## Code Change Summary

- Rotated existing underbody FK collision samples by root yaw and aligned their footprint to the diagnostic body samples.
- Raised root z from the max of root-center terrain support and yaw-rotated body-footprint support.
- Tuned only existing `fk_body_leg_collision` parameters in the MPC task cfg:
  - `weight = 640.0`
  - `shank_margin_m = 0.12`
  - `knee_margin_m = 0.04`
  - `underbody_margin_m = 0.05`
  - `shank_sample_count = 5`
- Kept `optimize_steps=24`.
- Did not add/delete/rename optimizer loss keys and did not change metric definitions.

## Key Metrics

| Scenario | Key Result |
| --- | --- |
| unit regression | `159 passed in 5.50s` |
| row8/col11, forward v0.50 x6 | `max_body_ground_penetration_count=0`, `max_fk_knee_ground_penetration_count=0`, `max_fk_shank_ground_penetration_count=0`, `min_fk_shank_ground_clearance_m=0.044592`, `max_terminal_planned_vs_fk_foot_error_m=3.815e-6` |
| row8/col12, forward v0.50 x6 | `max_body_ground_penetration_count=0`, `max_fk_knee_ground_penetration_count=0`, `max_fk_shank_ground_penetration_count=0`, `min_fk_shank_ground_clearance_m=0.044592`, `max_terminal_planned_vs_fk_foot_error_m=2.708e-6` |
| low-small baseline | exit `0`; `fk_foot_over_low_small_success_count=1`, diagonal `crossing_leg_count=1`, `fk_semantic_collision_count=0`, `fk_semantic_collision_rate=0`, `max_terminal_planned_vs_fk_foot_error=9.83e-7` |

## Result

- The farther-walk body/knee/shank heightfield penetration is cleared on row `8`, col `11` and row `8`, col `12`.
- FK planned-vs-realized readback remains micron-scale on both tiles.
- Old low-small hard metrics remain passing.

## Residual / Follow-Up

- Both rough-tile probes still report a small foot-ground residual near the initial frame: about `-0.00463m`.
- Touchdown-to-contact/current marker metrics can remain large on long multi-cycle sequences, especially row8/col11 (`max_touchdown_to_current_actual_foot_error_m=1.0732`). This was not fixed by FK body/leg collision tuning and should be treated as a separate marker/current-state semantics issue if reopened.

## Git Refs

- Baseline Ref: `8168b15` plus dirty working tree.
- Candidate Ref: `8168b15` plus dirty working tree.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
