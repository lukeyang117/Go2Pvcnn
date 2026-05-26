# T302h Rolling25 Low-Small Foot-Over Production

## Purpose

Align the semantic obstacle probe with runtime MPC behavior: single plan horizon `25`, repeated replans over a total `300` step rollout. Then productionize low-small foot-over loss changes so the foot crosses above low small obstacles instead of routing around them.

## Stage

- production `extension/batch_mpc_planner`
- probe/test `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
- related todo: [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Changes

- `low_small_foot_over_loss` now includes path-curve/window hooks and stronger low-small crossing height clearance.
- Production low-small foot-over defaults now use `clearance_m=0.065`, `z_weight=420`, `path_curve_weight=120`, `path_curve_z_weight=90`.
- `plan_segment` now replaces non-finite optimizer rows with state-grounded standstill fallback before returning, instead of returning NaN trajectories marked only by `safe_fallback`.
- Probe rolling logic now treats `--requested-n-frames 300` as total rollout length and forces each MPC segment to `25` frames before replanning from the current Isaac state/scanner.
- Low-small foot-over metric now uses the semantic object anchor footprint and object top height, instead of requiring `semantic_at(foot_xy)==small` from a stale initial local scanner over the full rolling trajectory.

## Commands

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 24 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline > tmp/t302h/production_true_rolling25_low_small_clearance065.jsonl 2>&1
pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/extension/batch_mpc_planner/optimizer.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
git diff --check -- Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/extension/batch_mpc_planner/optimizer.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

## Key Metrics

Final true rolling25 low-small production result:

- `semantic_task_violation=0/2`
- `small_overpass_success=2/2`
- `foot_over_low_small_success=2/2`
- stance/touchdown semantic contact: `0`
- `foot_semantic_penetration_rate=0`
- no non-finite rolling diagnostics
- forward: `foot_accel_max_to_mean=22.194`, `jump=11.185`, `boundary=7.340`, `R2=0.397`
- mixed/yaw: `foot_accel_max_to_mean=29.451`, `jump=5.805`, `boundary=8.422`, `R2=0.325`

Compared with the previous true rolling25 pass after NaN fallback but before clearance tuning:

- mixed/yaw `semantic_task_violation 1 -> 0`
- mixed/yaw `foot_semantic_penetration_rate 0.006667 -> 0`
- mixed/yaw `R2 0.104 -> 0.325`
- forward remained task-clean and `foot_accel_max_to_mean 25.174 -> 22.194`

## Verification

- backend: `110 passed, 1 warning`
- semantic probe unit tests: `53 passed`
- py_compile: pass
- diff check: pass

## Remaining Risk

- `min_z_quadratic_r2` remains moderate (`0.325-0.397`), so the foot is continuous enough for current gates but not visually perfect parabola quality.
- `playback_foot_error_max` remains around `0.29m` in the rolling probe, so planned-foot vs realized-foot mismatch is still a separate open issue.

## Git Refs

- Baseline Ref: working tree before rolling25/fallback/clearance pass
- Candidate Ref: working tree
- Key files:
  - `Go2Pvcnn/extension/batch_mpc_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/production_true_rolling25_low_small_clearance065.jsonl`
