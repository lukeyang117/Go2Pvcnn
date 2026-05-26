# T302h Local Overpass And Post-Blend Selector

## Purpose

Continue T302h test-only search until the user's semantic obstacle requirement is covered by metrics:

- low small obstacle: cross over it locally along the commanded path, without stance/touchdown/foot penetration, and with continuous foot/root trajectories
- high small obstacle: avoid/reroute like a high obstacle
- large obstacle: avoid/reroute with clearance, no semantic contact, and continuous trajectories

## Stage

- `extension/batch_mpc_planner` test-only semantic MPC diagnostics
- Probe files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`

## Related Todo

- [../todo/T302h-semantic-obstacle-jitter-reproduction.md](../todo/T302h-semantic-obstacle-jitter-reproduction.md)

## Procedure

All result artifacts were kept under `tmp/t302h/`.

Test-only directions tried:

- `command_path_*` metrics: body-frame command rollout path tube to distinguish true mixed-command detours from normal `vx/vy/yaw` curves.
- `ever_crossed_obstacle_along_command`: fixes long-horizon mixed-yaw cases where the root crosses over the obstacle and later turns back.
- local overpass metric: low-small crossing now requires local passage through the obstacle lane (`small_overpass_local_lateral <= lane_limit`), not global full-horizon path drift.
- `path_tube_low_small_task`: rejected as optimizer loss; it worsened path drift/root acceleration in the mixed low-small case.
- `select_policy_class_metric_task_jitter_margin`: metric-only selector; useful for low-small mixed but not enough for large-forward continuity.
- `post_blend_body_hard_contact_only`: probe-only result post-processing to test whether large-forward jump is a frame-0 handoff artifact.
- `select_policy_class_large_smooth_metric_margin`: final test-only selector combining class-aware candidate pools, metric ordering, and post-blend candidate.

## Key Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small,large --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline,select_policy_class_large_smooth_metric_margin > tmp/t302h/acceptance_best_selector_low_large_sweep.jsonl 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --cases small --cycles 1 --requested-n-frames 300 --playback-frame 299 --warmup-steps 6 --semantic-small-height-m 0.46 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,yaw100:0.00 0.00 1.00' --variants baseline,select_policy_class_large_smooth_metric_margin > tmp/t302h/acceptance_best_selector_high_small_sweep_v2.jsonl 2>&1
```

## Metrics

- `small_overpass_success`: low-small success. Requires linear command, local path crossing over the obstacle lane, no stance/touchdown semantic contact, no foot semantic penetration, and continuity within gate.
- `large_avoid_success`: high-small/large success. Requires semantic contact clean, root semantic clean, clearance margin satisfied, and continuity within gate.
- `semantic_task_violation`: task-level failure flag. This is the main pass/fail count.
- `semantic_task_contact_violation`: stance/touchdown/foot penetration failure.
- `semantic_task_continuity_violation`: foot/root acceleration, jump, or boundary discontinuity failure.
- `semantic_policy_margin_deficit`: missing avoidance distance for high-small/large. Lower is better; `0` passes.
- `worst_max_to_median_step`: worst swing-step jump ratio. Lower is smoother; task gate is `<=30`.
- `foot_accel_max_to_mean` / `root_accel_max_to_mean`: trajectory jitter ratios. Lower is smoother; task gate is `<=30`.

## Results

Default low-small + large, 3 commands each:

- Baseline: `semantic_task_violation=3/6`, `small_overpass_success=0`, `large_avoid_success=2/3`, `contact_violation=2`, `continuity_violation=1`, `max_jump=105.334`, `score_mean=639.675`.
- `select_policy_class_large_smooth_metric_margin`: `semantic_task_violation=0/6`, `small_overpass_success=2/2` for linear low-small commands, `large_avoid_success=3/3`, `contact_violation=0`, `continuity_violation=0`, `max_jump=15.290`, `score_mean=302.787`.
- Improvement vs baseline on the combined low-small/large sweep: task violations `3 -> 0`, score mean `-52.66%`, max jump `-85.48%`, max foot acceleration `31.099 -> 20.879`.

High-small (`semantic_small_height_m=0.46`), 3 commands:

- Baseline already passed task gate: `semantic_task_violation=0/3`, `large_avoid_success=3/3`, `max_jump=28.330`, `score_mean=525.391`.
- `select_policy_class_large_smooth_metric_margin`: `semantic_task_violation=0/3`, `large_avoid_success=3/3`, `small_overpass_success=0`, `contact_violation=0`, `continuity_violation=0`, `max_jump=5.305`, `score_mean=442.193`.
- Improvement vs baseline: score mean `-15.83%`, max jump `-81.27%`; high-small remains avoidance, not crossing.

Large forward root cause:

- Best non-blended large-forward candidate had clearance and contact clean but failed continuity because worst foot jump occurred at frame `0`.
- Probe-only post-blend candidate reduced large-forward jump to `8.837` in the selected sweep and produced `large_avoid_success=1`.

## Result

Pass for test-only direction search. The current best tested direction is `select_policy_class_large_smooth_metric_margin`, backed by real IsaacLab 300-step low-small, high-small, and large obstacle sweeps.

No production planner/runtime code was changed.

## Follow-Up

- Production implementation should not copy the probe post-processing blindly. The evidence says the remaining large-forward discontinuity is a handoff/first-frame foot trajectory issue, so the production fix should target planner/viewer handoff or first-frame foot blending/anchoring with contact semantics.
- Run a multi-cycle near-obstacle acceptance after production design, because current acceptance is single-cycle 300-step per command.

## Verification

- `pytest -q Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`: `28 passed`
- `py_compile`: pass
- `git diff --check`: pass
- `git diff -- Go2Pvcnn/extension/batch_mpc_planner/config.py | wc -l`: `0`

## Git Refs

- Baseline Ref: working tree before this test-only metric/selector pass
- Candidate Ref: working tree
- Key Files:
  - `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`
  - `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py`
  - `tmp/t302h/acceptance_best_selector_low_large_sweep.jsonl`
  - `tmp/t302h/acceptance_best_selector_high_small_sweep_v2.jsonl`
