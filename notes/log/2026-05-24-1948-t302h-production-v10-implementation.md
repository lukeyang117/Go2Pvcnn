# T302h Production V10 Implementation

## Purpose

Convert the best T302h test-only semantic obstacle direction into production `extension/batch_mpc_planner` code without selector/postprocess machinery.

Target behavior:

- low small obstacle: cross over, no stance/touchdown semantic contact, continuous root/foot trajectory
- high small obstacle: avoid, no semantic contact, continuous trajectory
- large obstacle: avoid, no semantic contact, continuous trajectory

## Stage

- production `extension/batch_mpc_planner`
- related todo: [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Code Changes

- Added `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`.
  - Classifies semantic corridor mode as low-small forward, low-small mixed/yaw, or high/large avoidance.
  - Shapes high-small/large planning commands by reducing forward velocity and adding lateral velocity toward the freer side.
- Updated `plan_segment`.
  - Builds and optimizes with an internal shaped planning command for high-small/large obstacles.
  - Keeps the external caller contract unchanged: caller still passes the requested command.
- Added low-small production losses:
  - `low_small_foot_crossing_loss`: rejects stance and touchdown on low small obstacle cells/soft field.
  - `low_small_stepcap_continuity_loss`: gates to low-small mixed/yaw commands and penalizes worst foot/root steps, acceleration, jerk, and first-frame foot drift.
- Added config fields and task-cfg overrides for the new low-small losses.
- Added production tests in `Go2Pvcnn/tests/test_batch_mpc_backend.py`.

## Important Implementation Finding

The first production pass shaped only the nominal seed while leaving optimizer/tracking losses on the original command. That improved low-small and high-small but left large-forward with a continuity failure:

- low/large first pass: `semantic_task_violation_count=1/4`
- large-forward: `worst_max_to_median_step=33.965`, `semantic_task_continuity_violation=1`

The root cause was that the optimizer was still being pulled toward the original forward command after the nominal seed had been laterally shaped. The final production behavior uses the shaped command consistently as the internal planning command for nominal and optimizer/loss terms when high-small/large avoidance is active.

## Verification Commands

```bash
pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py -k 'semantic_policy_routes or nominal_command_shaping or low_small_foot_crossing or low_small_stepcap'
pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git diff --check -- Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py Go2Pvcnn/tests/test_batch_mpc_backend.py
```

Real IsaacLab acceptance:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,forward_v050:0.50 0.00 0.00' --variants baseline > tmp/t302h/production_v10_low_large_sweep_v2.jsonl 2>&1
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline > tmp/t302h/production_v10_high_small_sweep_v2.jsonl 2>&1
```

## Key Metrics

Production low-small/large 4-row sweep:

- `semantic_task_violation_count=0/4`
- `small_overpass_success_count=2/2`
- `large_avoid_success_count=2/2`
- `contact_violation_count=0`
- `continuity_violation_count=0`
- score mean `217.941`
- max foot acceleration ratio `15.023`
- max root acceleration ratio `9.419`
- max jump ratio `8.468`

Production high-small `0.46m` 3-row sweep:

- `semantic_task_violation_count=0/3`
- `large_avoid_success_count=3/3`
- `contact_violation_count=0`
- `continuity_violation_count=0`
- score mean `276.147`
- max foot acceleration ratio `22.165`
- max root acceleration ratio `13.964`
- max jump ratio `7.032`

Compared with test-only v10 evidence:

- low/large score mean `218.304 -> 217.941`
- low/large task violations stay `0/4`
- high-small score mean `323.873 -> 276.147`
- high-small task violations stay `0/3`

## Result

Pass for the production implementation slice.

The production baseline now satisfies the corrected T302h task metrics in the targeted real IsaacLab 300-step sweeps.

## Follow-Up

- Broaden to multi-cycle near-obstacle replans.
- Run the larger T302/T300f non-regression set if this becomes the next training rollout baseline.
- Remaining visual risk: low-small rows still have moderate `min_z_quadratic_r2` in some runs; task/contact/continuity gates are clean, but visual parabolic swing shape may need a separate future loss if the user wants stricter shape aesthetics.

## Git Refs

- Baseline Ref: working tree before production implementation
- Candidate Ref: working tree
- Key output files:
  - `tmp/t302h/production_v10_low_large_sweep.jsonl`
  - `tmp/t302h/production_v10_high_small_sweep.jsonl`
  - `tmp/t302h/production_v10_low_large_sweep_v2.jsonl`
  - `tmp/t302h/production_v10_high_small_sweep_v2.jsonl`
