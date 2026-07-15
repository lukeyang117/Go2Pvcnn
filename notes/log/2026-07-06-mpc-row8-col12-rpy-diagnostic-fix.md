# MPC Row8 Col12 RPY Diagnostic Fix

## Purpose

Reproduce the user-reported existing `mpc` row `8`, col `12` planned-foot versus realized/FK-foot mismatch, check whether velocity/progress tracking is over-dominating the loss, and verify the 2026-05-28 low-small redesign metrics without touching the parallel `mpc_qp` work.

## Stage

- Existing MPC diagnostics and low-small acceptance probes.
- This log is for `planner_backend="mpc"` only, not `mpc_qp`.

## Related Todo

- [../todo/T302w-mpc-row8-col12-loss-tuning.md](../todo/T302w-mpc-row8-col12-loss-tuning.md)

## Commands

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py --device cuda:0 --terrain-row 8 --terrain-col 12 --commands 'forward_v050:0.50,0.00,0.00' --cycles 1 --requested-n-frames 25 --warmup-steps 4 --playback-frames 25
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants baseline --cycles 1 --requested-n-frames 25 --warmup-steps 6
```

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q -k root_rpy
```

## Input Conditions

- Terrain target: `--terrain-row 8 --terrain-col 12`.
- Backend: `planner_backend="mpc"`.
- Requested horizon: `25` frames.
- Plan dt: viewer/user command uses `0.02`.
- User hypothesis: velocity/progress tracking too strong, causing planned foot targets to drift away from realized/FK feet.

## Key Metrics

Initial diagnostic baseline before the RPY fix:

- `planned_fk_after_frame0_error_max_m ~= 0.188628`
- Temporary diagnostic-only `ik_fk x4 + kinematics x4`: apparent max error `~0.1497m`
- Temporary lowering of progress/swing authority: apparent max error stayed around `~0.1829m`

After extracting full roll, pitch, and yaw from `root_quat_w`:

- `planner_backend = "mpc"`
- `terrain_height_range_m = 0.4107666`
- `planned_fk_after_frame0_error_max_m = 2.7079e-6`
- `terminal_planned_vs_fk_foot_error_max = 2.7079e-6`
- `playback_readback_error_max_m = 4.266e-6`
- `speed_magnitude_tracking_error = 0.03075`
- `fk_semantic_collision_count = 0`

2026-05-28 low-small redesign acceptance probe:

- `cycle_count = 5`
- diagonal case `crossing_leg_count = 1`
- diagonal `fk_semantic_collision_count = 0`
- diagonal `fk_semantic_min_clearance_over_semantic_m = 0.1235`
- diagonal `planned_vs_fk_foot_error_crossing_leg_max_m = 9.7759e-7`
- summary `max_terminal_planned_vs_fk_foot_error = 9.7759e-7`

Broader tracked non-hard metrics:

- `max_fk_touchdown_on_small_rate = 0.25`
- `max_fk_stance_on_small_rate = 0.01`
- `max_fk_foot_small_penetration_rate = 0.02`

## Result

Pass for the hard T302w diagnosis:

- The apparent row `8`, col `12` planned-vs-FK mismatch was reproduced by the diagnostic path.
- Root cause was diagnostic-side RPY extraction: `_root_rpy_from_viewer_result()` fell back from `root_quat_w` to yaw only when `root_rpy` was absent, dropping roll/pitch on sloped terrain.
- The fallback now reconstructs full roll, pitch, and yaw from quaternion.
- A focused regression test covers the quaternion fallback and exits `0`: `1 passed, 57 deselected`.

## Conclusion

Do not tune production `mpc` losses based on the earlier `~0.188m` planned-vs-FK metric. After the diagnostic fix, the existing `mpc` row `8`, col `12` baseline has micron-scale planned-vs-FK/readback error, so the evidence does not support velocity/progress loss over-dominance as the root cause.

No new loss was added. No `mpc_qp` files or weights were modified for this branch.

## Follow-Up

- Keep [../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py](../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py) as a row-specific diagnostic guard.
- Treat the nonzero broader touchdown/stance-on-small rates as historical probe output unless the user explicitly makes those rates hard gates for this branch.

## Git Refs

- Baseline Ref: `8168b15`
- Candidate Ref: `8168b15` plus dirty working tree on 2026-07-06
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py](../../Go2Pvcnn/tests/mpc_row8_col12_loss_probe.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
