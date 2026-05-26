# T302i Reachable Crossing Probe Baseline

## Purpose

Create the new T302i low-small reachable-crossing probe requested by the user and establish baseline metrics for the current planner before changing losses.

## Stage

- `extension/batch_mpc_planner`
- semantic MPC viewer/runtime diagnostics
- related todo: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)
- design spec: [../../docs/superpowers/specs/2026-05-25-t302i-low-small-reachable-crossing-loss-design.md](../../docs/superpowers/specs/2026-05-25-t302i-low-small-reachable-crossing-loss-design.md)

## Command / Procedure

Local RED/GREEN:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
```

IsaacLab focused baseline:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_baseline.jsonl 2>&1
```

IsaacLab all-direction baseline:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'forward_v050:0.50 0.00 0.00,lateral_v050:0.00 0.50 0.00,diagonal_v050:0.35 0.35 0.00,mixed_yaw_v050:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_all_direction_baseline.jsonl 2>&1
```

## Input Conditions

- Python for real runs: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Device: `cuda:0`
- Case: low-small semantic obstacle
- Variants: `baseline`
- Total rollout: `300`
- Rolling replan horizon: `25`
- Warmup: `6`

## Key Metrics

Local checks:

- RED first failed on missing `mpc_low_small_reachable_crossing_probe`.
- Second RED failed on missing `reachable_swing_continuity_metrics`.
- Final local test: `5 passed`.
- `py_compile`: pass.

Forward baseline:

- `terminal_planned_vs_fk_foot_error_max=0.323130m`.
- `touchdown_ik_fk_error_max=0.682283m`.
- `fk_swing_foot_step_max_to_median=20.058457`.
- `fk_swing_foot_accel_max_to_mean=13.486868`.
- `fk_stance_on_small_rate=0`.
- `fk_touchdown_on_small_rate=0`.
- `fk_foot_small_penetration_rate=0`.
- `command_direction_cosine=0.999533`.
- `fk_foot_over_low_small_success=1`, `lift_then_land=1`, `touchdown_after=1`.

All-direction baseline:

| Command | Direction Cosine | Along Progress | Terminal Planned-vs-FK | Touchdown IK/FK | Swing Step Ratio | Swing Accel Ratio | Small Penetration | FK Foot-Over |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `forward_v050` | `0.999533` | `2.575964` | `0.323130` | `0.682283` | `20.058457` | `13.486868` | `0` | `1` |
| `lateral_v050` | `0.996041` | `2.283834` | `0.428282` | `0.656021` | `11.816412` | `7.203833` | `0` | `1` |
| `diagonal_v050` | `0.994865` | `1.074275` | `0.452422` | `1.063463` | `7.156092` | `5.672902` | `0` | `1` |
| `mixed_yaw_v050` | `-0.595472` | `-0.296376` | `0.388287` | `0.661772` | `12.329788` | `11.311711` | `0` | `1` |
| `yaw100` | `0` | `-0.180945` | `0.316823` | `0.399378` | `7.841658` | `6.735268` | `0.007500` | `0` |

## Result

Pass for creating the new probe and reproducing the current failure with the requested metrics.

The new probe starts IsaacLab, emits T302i-specific reachable-crossing metrics, and writes output under `tmp/t302i-viewer-realized-foot-mismatch/`.

## Conclusion

Current baseline still fails the new acceptance contract:

- Planned Cartesian feet/touchdowns remain far from FK-realized feet (`0.323-1.063m` depending metric/case).
- Swing continuity ratios are high.
- Mixed yaw direction tracking is wrong in the command-aligned metric (`command_direction_cosine=-0.595472`).
- Pure yaw still has small FK foot penetration (`0.0075`) even though crossing is correctly not required.
- Forward/lateral/diagonal FK-realized foot-over succeeds without small stance/touchdown contact, which means the next loss work should focus on feasibility/continuity/direction/stability rather than viewer rendering.

## Follow-Up

- Implement probe-only loss variants for reachable low-small crossing:
  - strengthen IK/FK and raw joint-limit feasibility;
  - add command-frame approach/cross direction gating;
  - add anti-low-base / anti-spider constraints;
  - preserve no-contact and FK foot-over.
- Re-run this same probe before production changes.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_forward_baseline.jsonl`
  - `../../tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_all_direction_baseline.jsonl`
