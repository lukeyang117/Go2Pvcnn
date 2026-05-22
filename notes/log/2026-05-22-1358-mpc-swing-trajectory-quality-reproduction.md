# 2026-05-22 13:58 MPC Swing Trajectory Quality Reproduction

## Purpose

Reproduce and quantify the user-reported MPC viewer problem where swing foot markers cluster at swing start, jump near swing end, and do not form a clean rise-then-fall arc.

## Stage

`extension/batch_mpc_planner` decoded foot trajectory quality and `extension/viz/go2_foostep_planner.py` viewer-runtime MPC path.

## Related Todo

- [T300 Unified Dense MPC Backend](../todo/T300-unified-dense-mpc-backend.md)

## Command / Procedure

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py
python -m py_compile Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py --device cuda:0 --terrain task --cycles 1 --commands forward --playback-frame 49 --requested-n-frames 50
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py --device cuda:0 --terrain task --cycles 2 --commands forward,yaw_left,forward_yaw_left --playback-frame 49 --requested-n-frames 50
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py --device cuda:0 --terrain task --cycles 1 --commands forward --playback-frame 49 --requested-n-frames 50 --trace-decode-layers
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py --device cuda:0 --terrain task --cycles 1 --commands yaw_left --playback-frame 49 --requested-n-frames 50 --trace-decode-layers
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py --device cuda:0 --terrain task --cycles 1 --commands forward,yaw_left --playback-frame 49 --requested-n-frames 50 --variants baseline,smooth8,smooth24,lr_half,lr_quarter,steps12,smooth8_lr_half,smooth24_lr_half,smooth8_steps12
```

## Input Conditions

- Real `env_isaacsim` IsaacLab headless runtime.
- Task: `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0`.
- Planner backend: `mpc`.
- Terrain: `task`.
- Horizon: `50` frames.
- Commands: `forward`, `yaw_left`, `forward_yaw_left`.

## Key Metrics

- Syntax checks:
  - `env_isaacsim` py_compile exit `0`
  - default Python py_compile exit `0`
- Single forward reproduction:
  - `worst_max_to_median_step=11.828948`
  - `worst_boundary_to_median_step=3.920286`
  - `worst_z_unimodal_violation_ratio=0.375`
  - `min_z_quadratic_r2=0.399121`
- Three-command, two-cycle matrix:
  - `cycle_count=6`
  - `max_worst_max_to_median_step=15.795617`
  - `max_worst_boundary_to_median_step=10.772456`
  - `max_worst_z_unimodal_violation_ratio=0.35`
  - `min_z_quadratic_r2=0.300820`
- Worst command/cycle examples:
  - `yaw_left` cycle `0`: `worst_max_to_median_step=15.795617`, `worst_boundary_to_median_step=10.772456`
  - `forward` cycle `0`: `worst_max_to_median_step=11.107587`, `worst_boundary_to_median_step=5.228242`
  - `forward_yaw_left` cycle `0`: `worst_max_to_median_step=10.788350`, `worst_boundary_to_median_step=5.037654`
- Decode-layer trace, `forward`:
  - `nominal`: `min_z_quadratic_r2=0.994759`, `worst_max_to_median_step≈2.13`, `worst_z_unimodal_violation_ratio=0.0`
  - `initial_decode_unlocked`: `min_z_quadratic_r2=0.993987`, `worst_max_to_median_step=2.127979`, `worst_z_unimodal_violation_ratio=0.0`
  - `initial_decode_locked`: `min_z_quadratic_r2=0.993987`, `worst_max_to_median_step=2.127979`, `worst_z_unimodal_violation_ratio=0.0`
  - `optimized_decode_unlocked`: `min_z_quadratic_r2=0.163466`, `worst_max_to_median_step=13.547757`, `worst_boundary_to_median_step=5.893077`
  - `optimized_decode_locked`: `min_z_quadratic_r2=0.363753`, `worst_max_to_median_step=13.547757`
- Decode-layer trace, `yaw_left`:
  - `initial_decode_locked`: `min_z_quadratic_r2=0.993987`, `worst_max_to_median_step=3.636983`, `worst_z_unimodal_violation_ratio=0.0`
  - `optimized_decode_unlocked`: `min_z_quadratic_r2=0.206992`, `worst_max_to_median_step=10.599597`, `worst_boundary_to_median_step=5.431489`
  - `optimized_decode_locked`: `min_z_quadratic_r2=0.381881`, `worst_max_to_median_step=11.065320`
- Test-only variant sweep, one cycle each for `forward,yaw_left`:
  - best by aggregate score: `smooth24`, `score_mean=12.851905`, `score_max=12.855396`
  - `smooth24`: `max_worst_max_to_median_step=5.248073`, `max_worst_boundary_to_median_step=1.565717`, `min_z_quadratic_r2=0.562506`
  - `smooth8`: `score_mean=14.997626`, `max_worst_max_to_median_step=7.341973`, `max_worst_boundary_to_median_step=5.429005`
  - `smooth24_lr_half`: higher `min_z_quadratic_r2=0.718175`, but worse jump ratio `18.087816`, so score is worse than plain `smooth24`
  - baseline in same sweep: `score_mean=30.021703`, `max_worst_max_to_median_step=16.202209`, `max_worst_boundary_to_median_step=8.982029`, `min_z_quadratic_r2=0.225205`
  - learning-rate-only variants were not enough: `lr_quarter score_mean=23.386617`, `lr_half score_mean=38.390182`

## Result

Pass as reproduction, instrumentation, and test-only direction screening.

## Conclusion

The new probe turns the screenshot symptom into repeatable numeric signals:

- Swing trajectories contain large local jumps relative to their normal frame-to-frame spacing.
- The contact/swing boundary can be much larger than the median in-swing step.
- The foot height profile is often not close to a simple parabolic rise/fall shape.

Root cause narrowed: the issue first appears after optimizer updates `foot_pos_residual`. It is not caused by marker rendering, and it is not first introduced by touchdown locking. `nominal` and initial decode are smooth/parabolic; `optimized_decode_unlocked` already has large local jumps and low quadratic fit before lock-to-touchdown postprocessing. Touchdown locking can still alter/flatten terminal stance frames, but it is secondary to the optimized residual shape problem.

The test-only sweep suggests the strongest first fix direction is stronger swing smoothness/residual shaping, not optimizer learning-rate reduction alone. `smooth24` cut the same-sweep aggregate score from `30.021703` to `12.851905`, reduced the worst jump ratio from `16.202209` to `5.248073`, and reduced the worst boundary ratio from `8.982029` to `1.565717`, while still leaving parabolic fit imperfect.

## Follow-Up

- Add a failing backend-level quality test for optimized swing smoothness/parabolic shape, ideally avoiding full IsaacLab startup.
- Candidate fix direction: add a residual/trajectory regularizer that preserves swing shape, such as second-difference foot smoothness on swing frames, nominal foot tracking on swing frames, or a direct swing-height/parabolic-shape loss. Use the `smooth24` sweep as a test-layer target signal, but do not copy it blindly into production defaults without checking T302 collision/obstacle regressions.
- Do not fix in `go2_foostep_planner.py`; current evidence points to optimizer/loss design.

## Git Refs

- Baseline Ref: working tree before adding the probe.
- Candidate Ref: working tree with [../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py](../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py).
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py](../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
