# MPC Semantic Obstacle Jitter Reproduction

## Purpose

Reproduce the user-reported root/foot shaking and discontinuous foot planning near semantic small/large objects under real IsaacLab, and add quantitative semantic-object collision rates alongside the previous T300f swing trajectory metrics.

## Stage

- `extension/batch_mpc_planner`
- semantic MPC viewer/runtime diagnostics
- T302/T302g semantic obstacle behavior
- T300f swing trajectory quality metrics

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)
- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)
- [../todo/T300-unified-dense-mpc-backend.md](../todo/T300-unified-dense-mpc-backend.md)

## Command / Procedure

RED helper test:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

Expected initial failure: `ModuleNotFoundError: No module named 'mpc_semantic_obstacle_jitter_probe'`.

GREEN helper tests:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

Real IsaacLab reproduction:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py \
  --device cuda:0 \
  --cases small,large \
  --cycles 1 \
  --requested-n-frames 300 \
  --playback-frame 299 \
  --warmup-steps 6
```

## Input Conditions

- One env.
- Semantic MPC task: `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0`.
- Semantic anchors: S4 `small` and S4 `large`.
- Start offset: `0.35m` before the target anchor along command direction.
- Commands:
  - `forward_v050 = (0.50, 0.00, 0.00)`
  - `forward_yaw_v050_vy025_yaw100 = (0.50, 0.25, 1.00)`
  - `yaw100 = (0.00, 0.00, 1.00)`
- Horizon: `300` frames.
- Playback frame for next-state handoff: `299`.

## Key Metrics

| Case | Key Result |
| --- | --- |
| `small + forward_v050` | `stance_on_small_rate=0.053571`, `foot_on_small_rate=0.037500`, `foot_accel_max_to_mean=30.885206`, `root_accel_max_to_mean=16.538178`, `worst_jump=5.045945`, `boundary=2.875158`, `R2=0.542522` |
| `small + forward_yaw_v050_vy025_yaw100` | `crossed_obstacle_along_command=1`, `min_root_distance_to_obstacle=0.252402`, `root_accel_max_to_mean=40.758645`, `foot_accel_max_to_mean=24.883740`, `boundary=6.393271`, `R2=0.906944` |
| `small + yaw100` | `root_accel_max_to_mean=2.328490`, `foot_accel_max_to_mean=6.354469`, `boundary=8.279829`, `R2=0.486751` |
| `large + forward_v050` | `crossed_obstacle_along_command=1`, `foot_on_large_rate=0.020000`, `swing_over_large_rate=0.039474`, `large_penetration_rate=0.000833`, `root_accel_max_to_mean=35.116199`, `root_step_max_to_median=58.634496`, `boundary=8.754951`, `R2=0.742023` |
| `large + forward_yaw_v050_vy025_yaw100` | `foot_on_semantic_rate=0.001667`, `swing_over_semantic_rate=0.003407`, `root_accel_max_to_mean=14.073759`, `foot_accel_max_to_mean=16.126863`, `worst_jump=23.649091`, `R2=0.907732` |
| `large + yaw100` | `foot_on_large_rate=0.032500`, `swing_over_large_rate=0.064784`, `large_penetration_rate=0.001667`, `foot_accel_max_to_mean=20.277184`, `R2=0.351219` |

Probe footer:

- `cycle_count=6`
- `max_stance_on_semantic_rate=0.053571`
- `max_touchdown_on_semantic_rate=0.0`
- `max_root_on_semantic_rate=0.0`
- `max_root_accel_max_to_mean=40.758645`
- `max_foot_accel_max_to_mean=30.885206`
- `worst_score=650.610166`

## Result

Pass as reproduction.

The new probe reproduces semantic-object contact and high root/foot acceleration spikes near both small and large semantic anchors. The result is not a production fix; it is an evidence harness for comparing test-only candidate directions.

## Conclusion

- The issue is measurable under real IsaacLab with 300-step MPC horizon.
- The previous T302 strict acceptance (`0.0` collision ratios) does not cover this exact long-horizon near-anchor semantic jitter condition.
- The strongest reproduction signals are:
  - small obstacle stance contact in `small + forward_v050`
  - large obstacle swing-over/penetration in `large + forward_v050` and `large + yaw100`
  - root acceleration spike ratio up to `40.758645`
  - foot acceleration spike ratio up to `30.885206`
- Next work should compare candidate directions inside this probe before production changes.

## Follow-Up

- T302h.2: run multi-cycle near-obstacle sequences to quantify repeated replan shake.
- T302h.3: add test-only candidate variants and rank by semantic collision + root/foot jitter + T300f swing shape metrics.

## Git Refs

- Baseline Ref: `working tree before T302h probe`
- Candidate Ref: `working tree with T302h probe`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
