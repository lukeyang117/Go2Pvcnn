# T302g Global-Sync Counter Cleanup

## Purpose

Finish the T302g global-sync cleanup by removing the last dirty-subset-style runtime counter names from the active MPC manager/tests and verify backend consistency without launching heavy IsaacLab runtime.

## Stage

T302g MPC manager scheduling / backend diagnostics naming cleanup.

## Related Todo

- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
  - `T302g.5c`
  - `T302g.7`

## Command / Procedure

- Residual-string scan:
  - `rg -n "dirty_subset|max_dirty_envs_per_step|replan_mode|selection_mode|fixed_topk_priority|dirty_backlog|selected_dirty_count|dirty_count" Go2Pvcnn/extension/batch_mpc_planner Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_runtime_headless.py`
- Backend regression:
  - `pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q`
- Syntax check:
  - `python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_runtime_headless.py`

## Input Conditions

- Baseline ref: working tree after the earlier global-sync sampled-MPC backend pass.
- Candidate ref: same working tree with counter-name cleanup only.
- No real IsaacLab runtime launched in this pass because the user requested minimal GPU usage.

## Key Metrics

- Residual scheduler scan after cleanup:
  - only historical guard assertion remains: `assert "mpc_max_dirty_envs_per_step" not in source`
- Backend regression:
  - `91 passed, 1 warning`
- `py_compile`:
  - exit `0`

## Result

Pass.

Active runtime counters for the T302g global-sync path now use global-sync semantics:

- `global_due`
- `global_due_count`
- `sampled_plan_count`
- `max_stale_observed`
- `planner_ms`
- `cache_ms`

Removed from active manager/test expectations:

- `dirty_count`
- `selected_dirty_count`
- `dirty_backlog`

## Conclusion

The code and tests now tell the same story as the current T302g design: planning participation is a sampled global-tick event, not a dirty-subset backlog scheduler.

## Follow-Up

- Real IsaacLab timing remains under [2026-05-18-1258-t302g-4096-timing-diagnostics.md](2026-05-18-1258-t302g-4096-timing-diagnostics.md).
- Full T302 strict JSONL non-regression is still open under `T302g.6`.

## Git Refs

- Baseline Ref: working tree after `2026-05-18 17:29` backend verification
- Candidate Ref: working tree after `2026-05-18 22:00` counter cleanup
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
