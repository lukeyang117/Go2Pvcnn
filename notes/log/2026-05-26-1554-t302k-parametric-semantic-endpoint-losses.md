# T302k Parametric Semantic And Endpoint Losses

## Purpose

Continue T302k.9 by adding parametric sampled-frame losses for high/large semantic avoidance, touchdown endpoint consistency, and swing foot height guarding, then verify local tests and IsaacLab probes.

## Stage

`extension/batch_mpc_planner` parametric MPC.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Code Changes

- Added sampled loss keys:
  - `parametric_semantic_avoidance`
  - `parametric_touchdown_endpoint`
  - `parametric_foot_height_guard`
- Routed parametric decode through existing semantic command shaping for high-small/large obstacles, while keeping original command for command-progress loss.
- Updated `parametric_v1` probe cfg helpers to request at least 40 optimization steps.

## Verification

Local:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "parametric_plan_shapes_root_laterally_around_high_large_obstacle"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

Results:

- Red first: high/large parametric root lateral offset was `0.0m`.
- Green after shaping: targeted test passed.
- Focused local suite: `220 passed, 1 warning`.

IsaacLab GPU3:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants parametric_v1 --requested-n-frames 300 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,pure_yaw:0.00 0.00 1.00' > tmp/t302k-parametric-mpc/low_small_parametric_v1_t302k9_opt40_gpu3.jsonl 2>&1
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --variants parametric_v1 --cases small,large --semantic-small-height-m 0.46 --requested-n-frames 300 --warmup-steps 6 > tmp/t302k-parametric-mpc/high_large_parametric_v1_t302k9_shaped_gpu3.jsonl 2>&1
```

## Key Metrics

Low-small:

- `fk_foot_over_low_small_success_count=3/3`.
- max FK stance/touchdown/small penetration rates all `0`.
- max terminal planned-vs-FK foot error `~1.9e-6m`.
- Endpoint issue remains: max touchdown IK/FK error `0.662m`.

High-small/large:

- `semantic_task_violation_count=6/6`.
- `large_avoid_success_count=0`.
- root semantic rate remains `0`, but large yaw still has stance/touchdown semantic contact.
- max semantic penetration rate `0.0217`.
- max stance-on-semantic `0.0475`, max touchdown-on-semantic `0.0417`.
- continuity/policy violations remain the dominant task failure.

## Result

Partial:

- Local parametric loss contract and high/large root-shaping test pass.
- Low-small crossing remains good under IsaacLab.
- High-small/large acceptance is still not met. The next issue is structural: root/foot parametric curve constraints and avoidance margins are not strong enough for rolling semantic task metrics, especially continuity and yaw-large foot contacts.

## Follow-Up

- Continue T302k.9 with explicit root-path clearance acceptance and foot/touchdown semantic exclusion in the parametric curve generator, not old dense residual tuning.
- Endpoint/touchdown quality remains open despite the new loss key.

## Git Refs

- Baseline Ref: `1b799cd`
- Candidate Ref: working tree on top of `1b799cd`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
