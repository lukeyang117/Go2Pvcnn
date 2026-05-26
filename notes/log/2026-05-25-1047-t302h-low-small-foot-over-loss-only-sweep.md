# T302h Low-Small Foot-Over Loss-Only Sweep

## Purpose

Test whether probe-only loss modifications can make swing feet pass over low small obstacles, while preserving zero contact/penetration and swing continuity.

## Stage

- `extension/batch_mpc_planner` test/probe loss injection only
- related todo: [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Changes

Test/probe only:

- Added sensor-pose-aware `_terrain_grid_world_xy_for_probe(...)`; the previous helper ignored `terrain.sensor_pos_w` and `terrain.sensor_yaw`, so probe losses could chase local-grid coordinates instead of world obstacle coordinates.
- Added low-small foot-over loss-only variants:
  - `loss_low_small_footover_v1`
  - `loss_low_small_footover_clear_v2`
  - `loss_low_small_footover_cont_v3`
  - `loss_low_small_footover_gate_v4`
  - `loss_low_small_footover_gate_cont_v5`
  - `loss_low_small_footover_gate_strong_v6`
  - `loss_low_small_footover_leg_v7`
  - `loss_low_small_footover_leg_cont_v8`
  - `loss_low_small_footover_wide_v9`
  - `loss_low_small_footover_wide_cont_v10`
  - `loss_low_small_footover_cap_v11`
  - `loss_low_small_footover_cap_strong_v12`
- Added unit coverage for side-foot detour loss and sensor-pose grid conversion.

No production planner/runtime code was changed.

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline,loss_low_small_footover_gate_v4,loss_low_small_footover_cap_v11,loss_low_small_footover_cap_strong_v12 > tmp/t302h/low_small_footover_cap_loss_sweep.jsonl 2>&1
```

Earlier sweep artifacts in the same pass:

- `tmp/t302h/low_small_footover_loss_sweep.jsonl`
- `tmp/t302h/low_small_footover_gate_loss_sweep.jsonl`
- `tmp/t302h/low_small_footover_gate_loss_sweep_sensorfix.jsonl`
- `tmp/t302h/low_small_footover_leg_loss_sweep.jsonl`
- `tmp/t302h/low_small_footover_wide_loss_sweep.jsonl`

## Key Metrics

Baseline low-small production behavior:

- `semantic_task=2/2`
- `foot_over_low_small_success=0/2`
- `foot_over_low_small_min_lateral_mean=0.105655m`
- contact/penetration clean: stance `0`, touchdown `0`, penetration `0`
- continuity clean: foot accel ratio mean `4.545`, jump mean `2.915`

Best foot-over result after fixing sensor-pose coordinates:

- `loss_low_small_footover_gate_v4`
- `foot_over_low_small_success=2/2`
- `foot_over_low_small_min_lateral_mean=0.037337m`
- `foot_over_low_small_frame_count_mean=10`
- contact/penetration stayed clean: stance `0`, touchdown `0`, penetration `0`
- continuity failed: foot accel ratio mean `57.133`, jump mean `12.365`
- `semantic_task` stayed `2/2` because continuity failed

Other tested directions:

- pre-sensor-fix `footover_v1/v2/v3`: no foot-over success; some variants pushed feet farther from the obstacle.
- `leg_v7/v8`: reduced jump in some rows but lost mixed-command foot-over and kept foot accel ratio high.
- `wide_v9/v10`: did not stabilize foot-over; continuity still failed.
- `cap_v11/v12`: hard step/accel caps reduced neither the ratio nor the task violation; foot-over fell to `1/2`.

## Conclusion

Loss-only probe testing found a real fix to the targeting bug: sensor-pose-aware grid coordinates are required before any semantic loss can be trusted.

After that fix, loss-only can force swing feet over the low-small obstacle, but the optimizer realizes it as sparse foot jumps. The current best loss-only direction solves the foot-over metric but not the continuity requirement. More scalar smoothness/cap weight makes the foot-over/continuity tradeoff worse rather than better.

## Follow-Up

Open problem: produce a continuous swing-over trajectory without selector/postprocess/nominal-before logic. The next loss-only hypothesis should not add more scalar smoothness; it should encode swing-window shape or foot-over timing directly, otherwise the optimizer keeps finding sparse jump solutions.

## Git Refs

- Baseline Ref: working tree production v10 deterministic replan phase
- Candidate Ref: working tree test/probe-only loss variants
- Key files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/low_small_footover_cap_loss_sweep.jsonl`
