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


## 300-Step Cobblestone Long-Horizon Sweep

Command pattern, each cobblestone subterrain in an independent IsaacLab process to avoid fixture close/reopen hangs:

```bash
for terrain_case in flat random_rough hf_pyramid_slope hf_pyramid_slope_inv boxes pyramid_stairs pyramid_stairs_inv; do
  CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py \
    --device cuda:0 --terrain cobblestone --terrain-cases "${terrain_case}" \
    --speed-grid --vx-values 0.0,0.5,1.0 --vy-values 0.0,0.25,0.5 --yaw-value 1.0 \
    --cycles 1 --requested-n-frames 300 --playback-frame 299 --variants baseline \
    > "/tmp/mpc_cobblestone_300_${terrain_case}.jsonl" 2>&1
done
```

Coverage:

- Horizon: `300` frames.
- Terrain cases: `flat`, `random_rough`, `hf_pyramid_slope`, `hf_pyramid_slope_inv`, `boxes`, `pyramid_stairs`, `pyramid_stairs_inv`.
- Velocity grid: `vx in {0.0,0.5,1.0}`, `vy in {0.0,0.25,0.5}`, `yaw=1.0`.
- Total cycle summaries: `63` (`7` terrain cases x `9` commands).
- The first attempt to run all terrain cases inside one Isaac process stalled after `flat`; the successful run used one process per terrain case.

Terrain ranking by `score_mean` worst first:

| Terrain | score_mean | score_max | max jump ratio | max boundary ratio | max z violation | min z R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `boxes` | `44.013797` | `76.440777` | `39.043692` | `29.212166` | `0.338028` | `0.016839` |
| `random_rough` | `43.667021` | `69.800200` | `35.768132` | `29.478476` | `0.352941` | `0.052593` |
| `pyramid_stairs` | `42.763040` | `61.674512` | `34.212045` | `29.708757` | `0.361702` | `0.034815` |
| `hf_pyramid_slope_inv` | `42.692781` | `60.178890` | `36.633492` | `28.151879` | `0.368794` | `0.059943` |
| `pyramid_stairs_inv` | `41.402490` | `59.006725` | `31.662263` | `21.857821` | `0.371622` | `0.096603` |
| `hf_pyramid_slope` | `40.975688` | `56.377782` | `29.355465` | `18.669241` | `0.376812` | `0.004749` |
| `flat` | `38.809249` | `46.529104` | `31.213968` | `15.855385` | `0.371212` | `0.023181` |

Worst speed/terrain cases by aggregate score:

| Terrain | Command | score | jump | boundary | z violation | min z R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `boxes` | `vx=0.50, vy=0.25, yaw=1.00` | `76.440777` | `39.043692` | `29.212166` | `0.235714` | `0.417222` |
| `random_rough` | `vx=0.00, vy=0.25, yaw=1.00` | `69.800200` | `29.000510` | `29.478476` | `0.321918` | `0.189796` |
| `pyramid_stairs` | `vx=0.50, vy=0.00, yaw=1.00` | `61.674512` | `22.705930` | `29.708757` | `0.342857` | `0.416875` |
| `hf_pyramid_slope_inv` | `vx=0.50, vy=0.25, yaw=1.00` | `60.178890` | `36.633492` | `14.236549` | `0.295775` | `0.364890` |
| `pyramid_stairs_inv` | `vx=0.50, vy=0.00, yaw=1.00` | `59.006725` | `24.871601` | `21.857821` | `0.331034` | `0.103304` |
| `hf_pyramid_slope` | `vx=0.50, vy=0.00, yaw=1.00` | `56.377782` | `29.355465` | `16.543931` | `0.282609` | `0.234770` |
| `flat` | `vx=0.50, vy=0.00, yaw=1.00` | `46.529104` | `31.213968` | `2.730047` | `0.281690` | `0.023181` |

Interpretation: 300-step long-horizon yaw-heavy commands remain problematic even after the user-side smoothness weight change. The largest failures are not only z parabolic fit; many worst cases are dominated by huge boundary or in-swing step ratios, especially `boxes`, `random_rough`, and stairs/slope inverse cases. Mid-speed `vx=0.5` is the most frequent worst command family, while `vx=1.0` cases often score better because the median in-swing displacement is larger and boundary/jump ratios are less extreme.


## Test-Only Regularizer Direction Sweep

User requested no production code changes. The probe now supports test-only optimizer-loss wrappers through variant suffixes such as `baseline__exp_boundary8+exp_accel8+exp_residual4`. These wrappers monkeypatch the optimizer loss only while the probe calls `_plan_viewer_trajectory`, then restore the original function. Production planner defaults are not changed.

Representative 300-step cobblestone sweep:

- Terrain cases: `boxes`, `random_rough`, `pyramid_stairs` (the three worst `score_mean` cases from the previous full 7-terrain sweep).
- Speed grid: `vx in {0.0,0.5,1.0}`, `vy in {0.0,0.25,0.5}`, `yaw=1.0`.
- Horizon: `300` frames.
- Variants:
  - `baseline`
  - `baseline__exp_boundary8`
  - `baseline__exp_accel8`
  - `baseline__exp_residual4`
  - `baseline__exp_parabola4`
  - `baseline__exp_boundary8+exp_accel8`
  - `baseline__exp_boundary8+exp_accel8+exp_residual4`
  - `baseline__exp_boundary8+exp_accel8+exp_parabola4`

Averaged across the three representative terrains, sorted by `score_mean`:

| Variant | score_mean | score improvement | score_max improvement | jump improvement | boundary improvement | R2 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline__exp_boundary8+exp_accel8+exp_residual4` | `18.256209` | `58.797%` | `41.168%` | `40.792%` | `62.441%` | `+0.186653` |
| `baseline__exp_boundary8+exp_accel8` | `22.012576` | `50.319%` | `34.166%` | `51.846%` | `41.149%` | `-0.018073` |
| `baseline__exp_boundary8+exp_accel8+exp_parabola4` | `22.305074` | `49.659%` | `43.125%` | `60.728%` | `49.935%` | `-0.014941` |
| `baseline__exp_accel8` | `26.020740` | `41.273%` | `21.975%` | `22.657%` | `56.312%` | `-0.002766` |
| `baseline__exp_boundary8` | `29.959310` | `32.384%` | `26.785%` | `39.197%` | `32.535%` | `+0.037364` |
| `baseline__exp_residual4` | `33.082124` | `25.336%` | `11.330%` | `9.448%` | `32.704%` | `+0.067713` |
| `baseline` | `44.308045` | `0.000%` | `0.000%` | `0.000%` | `0.000%` | `0.000000` |
| `baseline__exp_parabola4` | `44.901892` | `-1.340%` | `-0.605%` | `-7.617%` | `3.984%` | `-0.057480` |

Per-terrain winners:

| Terrain | Best variant | Baseline score_mean | Best score_mean | Improvement |
| --- | --- | ---: | ---: | ---: |
| `boxes` | `baseline__exp_boundary8+exp_accel8+exp_residual4` | `43.765203` | `15.324605` | `64.987%` |
| `random_rough` | `baseline__exp_boundary8+exp_accel8+exp_residual4` | `45.492339` | `20.915643` | `54.025%` |
| `pyramid_stairs` | `baseline__exp_boundary8+exp_accel8+exp_residual4` | `43.666592` | `18.528378` | `57.568%` |

Interpretation:

- Best direction so far is a combined test-only regularizer: boundary continuity + second-difference foot acceleration + residual magnitude.
- Boundary-only and acceleration-only both help, but each has tradeoffs: boundary-only does not reduce jump enough; acceleration-only can leave or worsen boundary jumps.
- Residual-only helps but is too weak alone; it becomes valuable when paired with boundary+accel because it improves R2 and prevents the optimized residual from drifting too far away from nominal.
- The current direct parabola loss is not a good standalone direction and slightly worsened the averaged representative sweep. It should not be the first production change in this form.


## Production Boundary + Acceleration Regularizer

User selected only the `boundary8` and `accel8` directions for production, explicitly rejecting residual and parabola constraints because nominal should remain only a reference and its height lacks reliable elevation information.

Implemented production loss:

- `foot_boundary_smoothness_loss`: weighted mean of `||foot[t+1]-foot[t]||` near swing/contact probability transitions.
- `foot_acceleration_smoothness_loss`: weighted mean of second difference `||foot[t+1]-2*foot[t]+foot[t-1]||` on swing frames.
- New config: `losses.foot_trajectory_regularization`, defaults `weight=1.0`, `boundary_weight=8.0`, `accel_weight=8.0`.
- No residual-to-nominal loss and no parabola height loss were added.

Focused production verification:

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/losses/smoothness.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/losses/smoothness.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py
pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py --device cuda:0 --terrain cobblestone --terrain-cases boxes --commands 'grid_vx0.50_vy0.25_yaw1.00:0.5 0.25 1.0' --cycles 1 --requested-n-frames 300 --playback-frame 299 --variants baseline
```

Results:

- Backend: `99 passed, 1 warning`.
- Worst prior representative case before production regularizer, `boxes vx=0.50 vy=0.25 yaw=1.00`: `score=76.440777`, `jump=39.043692`, `boundary=29.212166`, `R2=0.417222`.
- Same focused case after production boundary+accel regularizer: `score=20.870477`, `jump=4.720598`, `boundary=8.645891`, `R2=0.530423`, `z_violation=0.280822`.
- Focused improvement: score `72.695%`, jump `87.909%`, boundary `70.405%`, R2 `+0.113201`.

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
