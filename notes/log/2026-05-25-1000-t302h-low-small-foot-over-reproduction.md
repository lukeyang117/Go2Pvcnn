# T302h Low-Small Foot-Over Reproduction

## Purpose

Reproduce the user's clarified low-small failure: the root locally crosses the small obstacle, but the feet go around the object instead of swinging over it.

## Stage

- `extension/batch_mpc_planner` test/probe metrics only
- related todo: [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Change

Test/probe only:

- Added `foot_over_low_small_*` metrics to `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`.
- Updated corrected low-small task gate so low-small overpass requires `foot_over_low_small_success=1`.
- Added unit coverage distinguishing a true swing-foot-over-small pass from a side detour.

No production planner/loss code was changed.

## Metric Definition

`foot_over_low_small_success=1` means at least one swing foot sample satisfies all of:

- foot XY samples the semantic small obstacle cell
- foot XY is within the local obstacle footprint threshold
- foot z is above the local terrain and above the semantic object height plus clearance margin

This separates "root crossed the obstacle lane" from "a foot actually swung over the object".

## Command

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00' --variants baseline > tmp/t302h/production_v10_low_small_foot_over_repro.jsonl 2>&1
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

## Key Metrics

Real IsaacLab low-small production baseline after adding the stricter foot-over metric:

- rows: `2`
- `semantic_task_violation_count=2`
- `small_overpass_success_count=0`
- `foot_over_low_small_success=0/2`
- `foot_over_low_small_rate=0.0`
- `foot_over_low_small_frame_count=0`
- nearest foot-to-obstacle-center lateral distances: about `0.106m` and `0.105m`
- contact remains clean: stance `0`, touchdown `0`, penetration `0`
- continuity remains clean: max jump `4.222`, max foot accel ratio `6.692`

Interpretation: the previous task gate counted root/local lane passage as low-small success, but the stricter foot-over metric shows the feet are avoiding the small object laterally instead of swinging over it.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `48 passed`
- `py_compile`: pass

## Result

Pass as reproduction.

The failure is now quantified and repeatable in `tmp/t302h/production_v10_low_small_foot_over_repro.jsonl`.

## Follow-Up

Next work should test loss-only directions that require swing-foot overpass for low-small obstacles while preserving:

- no stance/touchdown contact
- no foot penetration
- bounded foot/root continuity
- high-small and large avoidance behavior

## Git Refs

- Baseline Ref: working tree production v10 deterministic replan phase
- Candidate Ref: working tree test/probe metric changes
- Key files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/production_v10_low_small_foot_over_repro.jsonl`
