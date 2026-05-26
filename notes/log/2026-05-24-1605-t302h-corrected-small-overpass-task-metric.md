# T302h Corrected Small-Overpass Task Metric

## Purpose

Continue T302h with the user-corrected requirement: low small obstacles count as crossed only when the root follows the commanded path over the obstacle, stance/touchdown/foot penetration stay off semantic objects, and trajectory continuity remains bounded. High-small and large obstacles should be avoided by clearance/contact/continuity metrics.

## Stage

- `extension/batch_mpc_planner` test-only semantic MPC diagnostics
- Probe files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Procedure

- Added task-level diagnostics:
  - `small_overpass_success`
  - `large_avoid_success`
  - `semantic_task_violation`
  - task contact and continuity violations
- Corrected small-overpass semantics:
  - `crossed_obstacle_along_command` alone is not sufficient.
  - Low-small crossing allows `root_on_semantic_rate > 0` because the root can pass above the small obstacle.
  - Stance/touchdown semantic contact and foot penetration still fail crossing.
  - Lateral detours fail via `root_lateral_deviation_from_start_max`; `root_along_reverse_rate` remains diagnostic/sorting signal, not a hard gate.
- Added test-only candidates:
  - `straight_low_small_task`
  - `straight_smooth_low_small_task`
  - `select_policy_class_straight_task_jitter_margin`
- Kept all result artifacts under `tmp/t302h/`; no production planner config changes.

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --variants baseline,select_policy_class_clearance_jitter_margin,select_policy_class_straight_task_jitter_margin > tmp/t302h/body_yaw0_low_small_selector_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --variants baseline,select_policy_class_clearance_jitter_margin,select_policy_class_straight_task_jitter_margin > tmp/t302h/yaw_aligned_high_small_selector_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --variants baseline,select_policy_class_clearance_jitter_margin,select_policy_class_straight_task_jitter_margin > tmp/t302h/yaw_aligned_large_selector_sweep.jsonl 2>&1
```

## Key Metrics

Low-small, body-yaw0 selector sweep:

- `select_policy_class_straight_task_jitter_margin`: best score mean `278.999`.
- Low-small forward succeeded: `small_overpass_success=1`, `stance/touchdown/penetration=0`, `foot_accel=9.952`, `root_accel=5.383`.
- Low-small mixed `vx=0.50, vy=0.25, yaw=1.00` still failed: no contact, but `root_lateral_deviation_from_start_max=1.122` and `worst_max_to_median_step=15.290`.

High-small selector sweep:

- `select_policy_class_straight_task_jitter_margin`: `semantic_task_violation_count=0/3`, `large_avoid_success_count=3/3`.
- Contact stayed clean: max stance/root semantic contact `0`.
- Continuity stayed within task gate: max `foot_accel=15.986`, max `root_accel=18.277`.

Large selector sweep:

- `select_policy_class_straight_task_jitter_margin`: best score mean `364.969`, `large_avoid_success_count=2/3`.
- Large forward remained open: `semantic_policy_margin_deficit=0.0176`, `worst_max_to_median_step=86.445`, although foot/root accel improved versus clearance selector.

## Result

Partial pass with corrected metrics and clearer remaining failures.

- The previous metric was too weak because it allowed reaching the obstacle backside by detour.
- It was also too strict/wrong in one place because it treated `root_on_semantic_rate` as failure for low-small crossing; root passing above a small obstacle is allowed.
- The current best test-only selector improves task semantics and high-small avoidance, but it is not production-ready.

## Conclusion

Next useful direction is not more generic scalar smoothness. The unresolved failures are class/command-specific:

- low-small mixed forward+lateral+yaw needs a path-frame metric/loss that handles body-frame `vx,vy,yaw` without classifying commanded lateral motion as detour.
- large forward needs stronger avoidance clearance without introducing jump spikes.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `21 passed`
- `py_compile`: pass
- `git diff --check`: pass
- `git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l`: `0`

## Git Refs

- Baseline Ref: working tree before this test-only metric correction
- Candidate Ref: working tree
- Key Files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/body_yaw0_low_small_selector_sweep.jsonl`
  - `tmp/t302h/yaw_aligned_high_small_selector_sweep.jsonl`
  - `tmp/t302h/yaw_aligned_large_selector_sweep.jsonl`
