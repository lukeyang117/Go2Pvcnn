# T302h Seeded Diagnostics And V10 Sweep

## Purpose

Continue the semantic-obstacle jitter investigation in probe/test code only. The goal was to remove misleading run-order variance, diagnose foot/root acceleration spikes by frame/leg, and test whether a loss/nominal-before combination can satisfy:

- low small obstacle: cross over it, no stance/touchdown/penetration, continuous trajectory
- high small obstacle: avoid, no semantic contact, continuous trajectory
- large obstacle: avoid, no semantic contact, continuous trajectory

## Stage

- `extension/batch_mpc_planner` test-only semantic MPC diagnostics
- Key files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Changes In Test Code

- Added `_jitter_metrics` diagnostics:
  - `worst_foot_accel_frame`
  - `worst_foot_accel_leg`
  - `worst_foot_accel_value`
  - `worst_root_accel_frame`
  - `worst_root_accel_value`
  - `foot_accel_mean_for_ratio`
  - `root_accel_mean_for_ratio`
- Added deterministic probe seed per semantic class, command, cycle, and effective candidate. This prevents random nominal phase from making the same effective candidate appear different under different display variants.
- Added test-only variants:
  - `loss_low_small_stepcap_v3`
  - `loss_low_small_stepcap_v4`
  - `nominal_cmd_shape_a_combined_v9`
  - `nominal_cmd_shape_a_combined_v10`

## Commands

Local verification:

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
git diff --check -- Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py notes/todo.md notes/todo/T302h-semantic-obstacle-jitter-reproduction.md notes/log/index.md
git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l
```

Real IsaacLab sweeps:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline,loss_low_small_cont_v2,loss_low_small_stepcap_v3,loss_low_small_stepcap_v4
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,forward_v050:0.50 0.00 0.00' --variants baseline,nominal_cmd_shape_a_combined_v10
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline,nominal_cmd_shape_a_combined_v10
```

## Metrics

- `semantic_task_violation`: main task failure flag.
- `small_overpass_success`: low-small success; local root overpass plus no stance/touchdown/foot penetration and bounded continuity.
- `large_avoid_success`: high-small/large success; clearance margin plus no contact/root occupancy and bounded continuity.
- `semantic_task_contact_violation`: stance/touchdown/penetration failure.
- `semantic_task_continuity_violation`: foot/root acceleration, jump, or boundary failure.
- `foot_accel_max_to_mean`: foot acceleration spike ratio. Lower is smoother; gate is `<=30`.
- `foot_accel_max`: absolute worst foot second difference in meters per frame step. This distinguishes a real spike from ratio inflation caused by a small mean.
- `worst_foot_accel_frame/leg/value`: location of the worst foot acceleration spike.
- `root_accel_max_to_mean`: root acceleration spike ratio.
- `worst_max_to_median_step`: worst T300f jump ratio.
- `worst_boundary_to_median_step`: swing boundary continuity ratio.

## Key Results

Seeded diagnostic finding:

- Before seeding, the same effective candidate could look different depending on variant order.
- After seeding, `loss_low_small_cont_v2` and `nominal_cmd_shape_a_combined_v8` produce identical low-small mixed rows when their `effective_candidate` is the same.
- Low-small mixed failure in v2 was a real mid-trajectory foot spike: `worst_foot_accel_frame=103`, `leg=0`, `value=0.252306`, `foot_accel_max_to_mean=40.402`.

Low-small mixed, `vx=0.50, vy=0.25, yaw=1.00`:

- Baseline: `semantic_task=1`, `small_overpass=0`, contact violation `1`, continuity violation `1`, score `945.496`, footacc ratio `18.570`, rootacc ratio `31.425`.
- `loss_low_small_cont_v2`: contact cleaned but continuity failed, score `874.988`, footacc ratio `40.402`.
- `loss_low_small_stepcap_v4`: passed with `semantic_task=0`, `small_overpass=1`, contact `0`, continuity `0`, score `234.565`, footacc ratio `17.109`, rootacc ratio `7.415`, worst foot accel `0.0624`.
- Improvement of v4 vs baseline: score `-75.2%`, rootacc `-76.4%`, jump `-70.8%`, boundary `-51.1%`, stance contact `0.0353 -> 0`.

Low-small pure forward:

- Baseline: `semantic_task=1`, stance `0.0135`, root-on `0.0367`, score `788.719`.
- `loss_low_small_stepcap_v4` regressed pure forward: score `1471.638`, footacc ratio `107.866`.
- `struct_lowfoot_cross_hard` passed: `semantic_task=0`, `small_overpass=1`, contact `0`, continuity `0`, score `63.825`, footacc ratio `4.287`, rootacc ratio `2.491`.

Combined v10 low-small/large sweep:

- Baseline across four rows: `semantic_task_violation_count=3/4`, `small_overpass=0/2`, `large_avoid=1/2`, contact violations `2`, continuity violations `2`, score mean `816.388`.
- `nominal_cmd_shape_a_combined_v10`: `semantic_task_violation_count=0/4`, `small_overpass=2/2`, `large_avoid=2/2`, contact violations `0`, continuity violations `0`, score mean `218.304`.
- Improvement v10 vs baseline: score mean `-73.3%`, task violations `3 -> 0`, small-overpass `0/2 -> 2/2`, large-avoid `1/2 -> 2/2`, footacc max ratio `30.679 -> 17.109`, rootacc max ratio `31.425 -> 24.199`, jump max `46.127 -> 5.254`, boundary max `5.680 -> 2.359`.

High-small `0.46m` sweep:

- Baseline: `semantic_task_violation_count=0/3`, `large_avoid=3/3`, score mean `449.305`, max footacc `25.502`, max rootacc `20.178`, max jump `28.329`.
- `nominal_cmd_shape_a_combined_v10`: `semantic_task_violation_count=0/3`, `large_avoid=3/3`, score mean `323.873`, max footacc `22.165`, max rootacc `14.820`, max jump `11.773`.
- Improvement v10 vs baseline: score mean `-27.9%`, max jump `-58.4%`, max rootacc `-26.6%`, max footacc `-13.1%`.

## Result

Pass for this test-only direction search.

`nominal_cmd_shape_a_combined_v10` is the best current test-only direction:

- low-small pure forward: `struct_lowfoot_cross_hard`
- low-small mixed/yaw: `loss_low_small_stepcap_v4`
- high-small and large: `nominal_cmd_shape_a_conservative_v4`

Production planner/runtime code remains unchanged.

## Follow-Up

- Do not claim production solved yet; v10 is still a probe-only routing of test loss directions.
- If productionizing, avoid copying selector/postprocess machinery. The likely production change is a small class/command-conditioned loss selection plus conservative nominal-before command shaping for high/large.
- Remaining risk: low-small successful rows can have low `min_z_quadratic_r2` (`0.354-0.624`), so foot trajectory shape should be visually inspected or improved after the semantic/continuity gate is stable.
- The high-small/large avoidance rows pass the task gate but still carry old `semantic_policy_violation` in some large bypass cases because that legacy metric over-penalizes passing the projection line. Use `semantic_task_violation` and `large_avoid_success` as the corrected task gate.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `46 passed`
- `py_compile`: pass
- `git diff --check`: pass
- `git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l`: `0`

## Git Refs

- Baseline Ref: `c54dc5c`
- Candidate Ref: working tree test/probe changes
- Key output files:
  - `tmp/t302h/seeded_low_small_stepcap_sweep.jsonl`
  - `tmp/t302h/seeded_low_small_forward_candidates_sweep.jsonl`
  - `tmp/t302h/seeded_combined_v10_low_large_sweep.jsonl`
  - `tmp/t302h/seeded_combined_v10_high_small_sweep.jsonl`
