# 2026-05-21 22:24 MPC Flat Forward Touchdown Height Probe

## Purpose

Check whether viewer-visible MPC touchdowns appearing airborne on flat ground can be explained by a mismatch between the actual viewer touchdown marker positions, MPC bilinear `height_at(...)` sampling, and the `semantic_height_scanner` height map.

## Stage

`extension/batch_mpc_planner` viewer/runtime diagnostics.

## Related Todo

- [T302g MPC Semantic RL Training Config](../todo/T302g-mpc-semantic-rl-training-config.md)

## Command / Procedure

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
python -m py_compile Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py
CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py --device cuda:0 --cycles 5 --playback-frame 49
CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py --device cuda:0 --cycles 3 --playback-frame 49 --speeds 0.10,0.30,0.50
CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py --device cuda:0 --cycles 1 --playback-frame 49 --speeds 0.30 --zero-after-forward-frame 20
```

## Input Conditions

- Real IsaacLab headless runtime through `env_isaacsim`.
- Task: `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0`.
- Terrain: flat.
- Planner backend: `mpc`.
- Scanner: `semantic_height_scanner`.
- Speeds: forward-only `vx=0.10/0.30/0.50`, `vy=0`, `yaw=0`.
- Each cycle plans, reads the exact viewer marker path `PlannerVisualizer._touchdown_markers_world(result)`, compares heights, applies playback to frame 49, refreshes scanner, and replans.

## Key Metrics

- Single-speed `vx=0.30`, 5 cycles:
  - `max_abs_td_minus_mpc=0.000000`
  - `max_abs_td_minus_scanner_nearest=0.000001`
  - `max_abs_mpc_minus_manual_bilinear=0.000000`
  - `max_abs_mpc_minus_scanner_nearest=0.000001`
- Multi-speed `vx=0.10/0.30/0.50`, 3 cycles each:
  - `max_abs_td_minus_mpc=0.000000`
  - `max_abs_td_minus_scanner_nearest=0.000001`
  - `max_abs_mpc_minus_manual_bilinear=0.000000`
  - `max_abs_mpc_minus_scanner_nearest=0.000001`
- Re-run using the exact viewer touchdown marker helper:
  - `PlannerVisualizer._touchdown_markers_world(result)` maps `[B,T,4,3]` `planned_touchdown_w` to `[:, 0]`.
  - `vx=0.10/0.30/0.50`, 3 cycles each, again produced `max_abs_td_minus_mpc=0.000000` and `max_abs_td_minus_scanner_nearest=0.000001`.
- Reproduction of visible airborne cuboids:
  - Procedure: plan/play forward `vx=0.30` to frame `20`, then read current IsaacLab robot state and trigger zero-command MPC replan.
  - Zero-command path profile: `plan.zero_command_standstill`, `iters=0`.
  - `viz_td_z=[+0.06443, +0.12734, +0.11847, -0.00000]`
  - `viz_td_minus_mpc=[+0.06443, +0.12734, +0.11847, +0.00000]`
  - `state_foot_minus_mpc=[+0.06443, +0.12734, +0.11847, +0.00000]`
  - `td_equals_state_foot_max_abs=0.000000`

## Result

Pass as diagnostic evidence.

## Conclusion

On flat ground, moving-command planned touchdowns are grounded and the actual viewer touchdown marker z exactly matches the MPC bilinear height sampler. This rules out bilinear-vs-scanner height mismatch, and also rules out a separate viewer-marker extraction mismatch, for the normal moving-command plan.

The visible airborne cuboid issue is reproduced by a command transition: after a forward plan is played to a swing frame, zero-command standstill replanning bypasses `decode_trajectory(...)` and returns `_standstill_result_from_state(...)`. That function exports `planned_touchdown_w = state.foot_pos`, so any currently airborne swing foot becomes an airborne touchdown marker. This matches the user screenshot.

The probe did expose a separate anomaly: in this runtime setup, `vx=0.10` and `vx=0.30` produced negative final `dx` in the printed plan summaries, while `vx=0.50` produced positive final `dx`. That is a command-direction/planner-output follow-up, separate from touchdown height grounding.

## Follow-Up

- Fix `_standstill_result_from_state(...)` or the zero-command path so standstill touchdown markers are grounded from `height_at(...)` instead of copying current airborne foot positions.
- Track the forward-command negative-`dx` anomaly separately if it reproduces outside this diagnostic script.

## Git Refs

- Baseline Ref: working tree after 2026-05-21 grounded touchdown decode changes.
- Candidate Ref: working tree with diagnostic probe only.
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py](../../Go2Pvcnn/tests/mpc_flat_touchdown_height_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
