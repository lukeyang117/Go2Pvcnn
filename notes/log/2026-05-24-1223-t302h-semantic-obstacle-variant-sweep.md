# T302h Semantic Obstacle Variant Sweep

## Purpose

Compare semantic-obstacle loss directions inside the T302h probe before changing production MPC defaults, then verify whether the best test-only direction is stable enough to keep as actual code.

## Stage

- `extension/batch_mpc_planner`
- real IsaacLab MPC semantic obstacle diagnostics
- T302h semantic small/large/high-small jitter and collision behavior

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)
- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Command / Procedure

Helper verification:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
python -m py_compile \
  Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py \
  Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py
```

Representative IsaacLab sweeps:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py \
  --device cuda:0 --cases small,large --cycles 1 \
  --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 \
  --variants baseline,contact_only_semantic,high_body_margin,risk_crossing,risk_contact_crossing

CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py \
  --device cuda:0 --cases small --cycles 1 \
  --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 \
  --semantic-small-height-m 0.46 \
  --variants baseline,body_stance_crossing,body_stance_crossing_smooth
```

## Input Conditions

- Real IsaacLab under `env_isaacsim`.
- One env, 300 planned frames.
- Commands: `forward_v050`, `forward_yaw_v050_vy025_yaw100`, `yaw100`.
- Semantic cases:
  - low small: default small height `0.16m`
  - large: default large height `0.55m`
  - high small: overridden small height `0.46m`

## Key Metrics

First refined low-small/large sweep:

| Variant | contact_sum | pen_max | root_max | stance_max | large_swing_max | large_min_dist | small_min_dist | jump_max | boundary_max | R2_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.062334 | 0.001667 | 0.003333 | 0.044118 | 0.018152 | 0.171605 | 0.114414 | 34.665 | 6.317 | 0.057 |
| contact_only_semantic | 0.211856 | 0.001667 | 0.036667 | 0.142636 | 0.064463 | 0.349999 | 0.036096 | 13.503 | 4.997 | 0.303 |
| high_body_margin | 0.019167 | 0.000833 | 0.016667 | 0.000000 | 0.027586 | 0.349999 | 0.048935 | 49.362 | 4.699 | 0.345 |
| risk_contact_crossing | 0.017337 | 0.000833 | 0.006667 | 0.008170 | 0.019967 | 0.201367 | 0.296130 | 68.319 | 18.407 | 0.326 |
| risk_crossing | 0.022362 | 0.000833 | 0.000000 | 0.009983 | 0.017212 | 0.345049 | 0.068694 | 38.179 | 4.473 | 0.050 |

Focused body/stance direction:

| Case | Variant | contact_sum | pen_max | root_max | stance_max | large_swing_max | min_dist | cross_count | jump_max | boundary_max | R2_min |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-small | baseline | 0.0102 | 0.0000 | 0.0000 | 0.0102 | 0.0000 | 0.082 | 1/3 | 8.47 | 7.59 | 0.555 |
| low-small | body_stance_crossing | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.173 | 2/3 | 6.32 | 7.42 | 0.084 |
| low-small | body_stance_crossing_smooth | 0.0351 | 0.0000 | 0.0000 | 0.0351 | 0.0000 | 0.296 | 1/3 | 5.36 | 8.61 | 0.730 |
| large | baseline | 0.0108 | 0.0008 | 0.0100 | 0.0000 | 0.0034 | 0.191 | 1/3 | 52.23 | 9.05 | 0.701 |
| large | body_stance_crossing | 0.0025 | 0.0008 | 0.0000 | 0.0017 | 0.0034 | 0.350 | 0/3 | 18.62 | 3.01 | 0.206 |
| large | body_stance_crossing_smooth | 0.0017 | 0.0008 | 0.0000 | 0.0000 | 0.0348 | 0.350 | 1/3 | 18.61 | 6.27 | 0.060 |
| high-small | baseline | 0.0067 | 0.0000 | 0.0067 | 0.0000 | 0.0000 | 0.024 | 1/3 | 13.36 | 3.74 | 0.457 |
| high-small | body_stance_crossing | 0.0008 | 0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.303 | 0/3 | 7.64 | 3.09 | 0.570 |
| high-small | body_stance_crossing_smooth | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.134 | 1/3 | 15.82 | 6.10 | 0.770 |

Production verification attempt after temporarily applying `body_stance_crossing` defaults failed acceptance:

| Case | contact_sum | pen_max | root_max | stance_max | min_dist | cross_count | jump_max | R2_min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low-small | 0.0690 | 0.0000 | 0.0000 | 0.0690 | 0.255 | 2/3 | 5.01 | 0.428 |
| large | 0.0017 | 0.0008 | 0.0000 | 0.0000 | 0.350 | 1/3 | 26.06 | 0.546 |
| high-small | 0.0067 | 0.0000 | 0.0067 | 0.0000 | 0.044 | 1/3 | 9.80 | 0.586 |

## Result

Partial pass for test infrastructure and direction comparison; production change rejected and reverted.

- The high-small fixture issue is fixed: `semantic_small_height_m` now works for both event-driven semantic viewer configs and `SemanticCourseTerrainImporter` MPC semantic configs.
- `body_stance_crossing` is the best current test-only direction on the focused sweep:
  - low-small stance/root/penetration contact went to `0.0` while still crossing in `2/3` commands
  - high-small did not cross and kept root/stance at `0.0`, with only one `0.0008` foot penetration sample
  - large min root distance improved to `0.350m`, with large swing-over at `0.0034`
- `body_stance_crossing_smooth` is not selected: it improves some shape metrics but reintroduces low-small stance contact and high-small crossing.
- A temporary production-default attempt did not reproduce the focused test-only win robustly, so the production config was reverted.

## Conclusion

Do not productionize a single scalar default change from these one-cycle sweeps yet. The evidence points toward a more structured loss split:

- keep low-small crossing reward separate from stance/touchdown rejection
- strengthen high/large body avoidance separately
- add a stricter acceptance gate for high-small/large root path and low-small stance contact
- repeat the best direction with multi-cycle or seeded runs before changing production defaults

## Follow-Up

- Add a follow-up child under T302h for robust multi-cycle candidate acceptance.
- Prefer `body_stance_crossing` as the next hypothesis, but require it to pass production-baseline verification before actual code changes remain.

## Git Refs

- Baseline Ref: `working tree before production-default attempt`
- Candidate Ref: `temporary body_stance_crossing defaults, reverted after failed production probe`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py](../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py)
