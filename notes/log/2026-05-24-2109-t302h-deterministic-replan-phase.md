# T302h Deterministic Replan Phase

## Purpose

Continue the T302h production v10 verification after multi-cycle probes exposed an intermittent large-forward replan boundary failure. The focused hypothesis was that randomized nominal replan phase can choose a different diagonal swing/stance pattern at the next segment, causing a frame-0 foot discontinuity after Isaac playback.

## Stage

- production `extension/batch_mpc_planner` runtime config
- related todo: [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Change

- Set `MpcRuntimeCfg.randomize_replan_phase=False` by default.
- Added task-level override `mpc_randomize_replan_phase` in `planner_cfg_from_task_cfg(...)`.
- Added probe-only `phase_fixed_probe` for future ablation when a task explicitly enables phase randomization.
- Added regression tests for the deterministic default and probe variant.

This is not a selector, postprocess, handoff blend, or new loss. It changes the nominal gait phase prior to be deterministic across replans.

## Key Evidence

Probe-only comparison, large obstacle forward, `cycles=6`, `playback_frame=20`:

- baseline before default change: `semantic_task=1/6`, `large_avoid=5/6`, contact `0`, continuity `1/6`
- `phase_fixed_probe`: `semantic_task=0/6`, `large_avoid=6/6`, contact `0`, continuity `0/6`
- score mean `414.295 -> 336.443` (`-18.8%`)
- max jump `52.376 -> 23.012` (`-56.1%`)
- max boundary `9.091 -> 3.012` (`-66.9%`)
- max root accel ratio `21.194 -> 9.172` (`-56.7%`)
- max foot accel ratio `23.956 -> 16.711` (`-30.3%`)

After production default change, large-forward baseline matches `phase_fixed_probe`:

- `semantic_task=0/6`
- `large_avoid=6/6`
- contact `0`
- continuity `0/6`
- score mean `336.443`
- max jump `23.012`

Single-cycle acceptance after default change:

- low-small/large: `semantic_task=0/4`, low-small overpass `2/2`, large avoid `2/2`, contact `0`, continuity `0`
- high-small `0.46m`: `semantic_task=0/3`, large/high avoid `3/3`, contact `0`, continuity `0`

## Commands

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases large --cycles 6 --requested-n-frames 300 --playback-frame 20 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline,phase_fixed_probe > tmp/t302h/production_v10_large_forward_phase_fixed_pb20.jsonl 2>&1
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases large --cycles 6 --requested-n-frames 300 --playback-frame 20 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00' --variants baseline > tmp/t302h/production_v10_large_forward_phase_default_pb20.jsonl 2>&1
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,forward_v050:0.50 0.00 0.00' --variants baseline > tmp/t302h/production_v10_phase_default_low_large_sweep.jsonl 2>&1
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline > tmp/t302h/production_v10_phase_default_high_small_sweep.jsonl 2>&1
pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git diff --check -- Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_batch_mpc_backend.py notes/todo.md notes/todo/T302h-semantic-obstacle-jitter-reproduction.md notes/log/index.md
```

## Verification

- `pytest -q Go2Pvcnn/tests/test_batch_mpc_backend.py`: `104 passed, 1 warning`
- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `46 passed`
- `py_compile`: pass
- `git diff --check`: pass

## Remaining Risk

The long playback diagnostic still shows high `playback_foot_error_max` in some frame-299 rows. That means planned foot targets and realized IK/FK foot positions can still diverge over long playback. The deterministic phase change fixes the measured multi-cycle large-forward task/continuity failure, but it does not fully solve the separate planned-foot vs realized-foot mismatch.

## Git Refs

- Baseline Ref: working tree production v10 before deterministic phase default
- Candidate Ref: working tree
- Key files:
  - `Go2Pvcnn/extension/batch_mpc_planner/config.py`
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - `tmp/t302h/production_v10_large_forward_phase_fixed_pb20.jsonl`
  - `tmp/t302h/production_v10_large_forward_phase_default_pb20.jsonl`
  - `tmp/t302h/production_v10_phase_default_low_large_sweep.jsonl`
  - `tmp/t302h/production_v10_phase_default_high_small_sweep.jsonl`
