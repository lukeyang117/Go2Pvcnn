# T302 Strict Collision Metric Tuning Final Verification

## Purpose

Record the metric-driven MPC tuning outcome after the user asked to continue testing from numeric metrics until the acceptance requirements are met: complex-terrain traversability, no root/foot/knee/shank collision, low-small obstacle crossing, high-small/large obstacle avoidance or tracking deweighting, and no stance/touchdown on semantic obstacles.

## Stage

Production `Go2Pvcnn/extension/batch_mpc_planner` loss/config tuning plus real IsaacLab headless strict metric verification.

## Related Todo

- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Baseline Ref

- Working tree on top of `769f7d4`

## Candidate Ref

- Working tree on top of `769f7d4`

## Key Files

- [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
- [../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py](../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py)
- [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)

## Commands

```bash
TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q

TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:0 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
  # single-process strict JSONL probes for:
  # low small, high small, large, cobblestone
PY

TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp MPC_RUNTIME_HEADLESS=1 MPC_TEST_DEVICE=cuda:0 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_command_matrix_tracks_motion_and_limits_drift -q

TMPDIR=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/tmp \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/batch_mpc_planner/terrain.py \
  Go2Pvcnn/extension/semantic_course.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py

git diff --check
```

## Input Conditions

- Python env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Device: `cuda:0`
- Test output directory: [../../tmp/t302_mpc_metric_tuning/](../../tmp/t302_mpc_metric_tuning/)
- Strict metric cases:
  - flat low-small: forward, backward, lateral left, lateral right
  - flat high-small: forward
  - flat large: forward, yaw-left
  - COBBLESTONE: forward, backward, lateral left/right, yaw left/right, forward+yaw left/right, diagonal forward left/right

## Root Cause And Tuning

- Earlier strict metrics exposed residual high-small/large leg collision after the framework-level implementation:
  - `leg_collision.weight=6.0` left high-small knee collision near `0.005`.
  - `leg_collision.weight=8.0` fixed large forward strict collision but high-small still had a knee collision sample, with `knee_min_clearance=-0.00285m`.
- The final selected intersection was:
  - `MpcLossesCfg.leg_collision.weight=16.0`
  - `MpcLegCollisionLossCfg.knee_margin_m=0.06`
  - `MpcLegCollisionLossCfg.shank_margin_m=0.06`
  - `MpcLegCollisionLossCfg.worst_deficit_weight=16.0`
  - `shank_sample_count=2`
- TDD artifacts:
  - [../../tmp/t302_mpc_metric_tuning/red_default_leg_collision_w16_m06_worst16.txt](../../tmp/t302_mpc_metric_tuning/red_default_leg_collision_w16_m06_worst16.txt)
  - [../../tmp/t302_mpc_metric_tuning/green_default_leg_collision_w16_m06_worst16.txt](../../tmp/t302_mpc_metric_tuning/green_default_leg_collision_w16_m06_worst16.txt)

## Key Metrics

- Backend suite:
  - [../../tmp/t302_mpc_metric_tuning/backend_full_fresh_20260517_075614.txt](../../tmp/t302_mpc_metric_tuning/backend_full_fresh_20260517_075614.txt)
  - `75 passed in 4.69s`
- Fresh strict real IsaacLab JSONL aggregation:
  - [../../tmp/t302_mpc_metric_tuning/strict_low_small_fresh_20260517.jsonl](../../tmp/t302_mpc_metric_tuning/strict_low_small_fresh_20260517.jsonl)
  - [../../tmp/t302_mpc_metric_tuning/strict_high_small_fresh_20260517.jsonl](../../tmp/t302_mpc_metric_tuning/strict_high_small_fresh_20260517.jsonl)
  - [../../tmp/t302_mpc_metric_tuning/strict_large_fresh_20260517.jsonl](../../tmp/t302_mpc_metric_tuning/strict_large_fresh_20260517.jsonl)
  - [../../tmp/t302_mpc_metric_tuning/strict_cobblestone_fresh_20260517.jsonl](../../tmp/t302_mpc_metric_tuning/strict_cobblestone_fresh_20260517.jsonl)
  - `17` rows, `0` failed.
  - `max_root_bottom_collision_ratio=0.0`
  - `max_swing_foot_collision_ratio=0.0`
  - `max_knee_collision_ratio=0.0`
  - `max_shank_collision_ratio=0.0`
  - `max_stance_semantic_count=0`
  - `min_root_bottom_clearance=0.11492849886417389m`
  - `min_swing_foot_clearance=0.039048731327056885m`
  - `min_knee_clearance=0.05200812220573425m`
  - `min_shank_clearance=0.0067091286182403564m`
- Low-small acceptance:
  - forward/backward/lateral-left/lateral-right all crossed.
  - stance semantic count remained `0`.
- High-small acceptance:
  - forward did not cross.
  - `risk_linear_scale=0.5`.
  - `min_dist=0.2875557839870453m`, required `0.14m`.
- Large obstacle acceptance:
  - forward `min_dist=0.3142874836921692m`, required `0.305m`.
  - forward `risk_linear_scale=0.5`.
  - yaw-left `risk_yaw_scale=0.5`.
- COBBLESTONE acceptance:
  - `10` command rows, `0` failed.
  - all root-bottom/swing-foot/knee/shank collision ratios were `0.0`.
  - worst clearances: swing foot `0.039048731327056885m`, shank `0.029836177825927734m`.
- T300e regression:
  - [../../tmp/t302_mpc_metric_tuning/t300e_command_matrix_fresh_20260517.txt](../../tmp/t302_mpc_metric_tuning/t300e_command_matrix_fresh_20260517.txt)
  - `1 passed in 142.85s`
- Static:
  - [../../tmp/t302_mpc_metric_tuning/py_compile_fresh_20260517.txt](../../tmp/t302_mpc_metric_tuning/py_compile_fresh_20260517.txt), exit `0`
  - [../../tmp/t302_mpc_metric_tuning/git_diff_check_fresh_20260517.txt](../../tmp/t302_mpc_metric_tuning/git_diff_check_fresh_20260517.txt), exit `0`

## Result

Pass.

## Conclusion

The acceptance requirement is met for the tested real IsaacLab headless metric matrix: complex COBBLESTONE terrain remains traversable under the command set, flat low-small obstacles are crossed without stance on semantic obstacles, high-small and large obstacles are deweighted/avoided, and root-bottom, swing foot, knee, and shank collision ratios are all zero.

## Follow-Up

- The full-file `test_mpc_body_leg_collision_headless.py -q` path previously hung after one dot because of IsaacLab fixture reuse. The stronger evidence is the single-process strict JSONL probes above, which avoid fixture reuse and record per-case numeric metrics.
- Longer command-switch/yaw sequences and 4096-scale runtime counters remain future confidence layers if T302 is promoted into full RL rollout.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: working tree on top of `769f7d4`
