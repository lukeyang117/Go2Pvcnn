# T302g Global-Sync Sampled MPC Planning

## Purpose

Implement and verify the new T302g direction: fixed-interval `global_sync` MPC refresh with random `mpc_parallel_plan_batch_size` environment sampling, sampled-only MPC imitation reward, and hard reward-mask invalidation on reset/command changes.

## Stage

T302g MPC semantic RL training config / `extension/batch_mpc_planner` manager scheduling.

## Related Todo

- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)
  - `T302g.5a`
  - `T302g.5b`
  - `T302g.5c`

## Command / Procedure

- RED targeted backend test:
  - `pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "global_sync"`
- GREEN targeted backend test:
  - `pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "global_sync"`
- Broader backend regression:
  - `pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q`
- Syntax check:
  - `python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_runtime_headless.py`

## Input Conditions

- Baseline ref: `446a875`
- Candidate ref: working tree on top of `446a875`
- No real IsaacLab 4096 runtime was launched in this pass.
- T302g semantic task config now selects `mpc_replan_mode="global_sync"` and `mpc_parallel_plan_batch_size=4096`.

## Key Metrics

- RED test failed as expected before implementation:
  - `test_mpc_global_sync_does_not_replan_unsampled_or_command_changed_rows_before_interval`
  - old manager replanned before the global interval because unsampled rows kept stale per-env age.
- Targeted global-sync backend after fix:
  - `6 passed, 85 deselected`
- T302g/global-sync/task-config focused backend:
  - `7 passed, 82 deselected`
- Full backend:
  - `91 passed, 1 warning`
- `py_compile`:
  - exit `0`

## Result

Pass for backend behavior.

Implemented:

- `MpcRuntimeCfg.replan_mode`, defaulting to legacy `"dirty_subset"` for non-T302g callers.
- `MpcRuntimeCfg.parallel_plan_batch_size`, mapped from task cfg `mpc_parallel_plan_batch_size`.
- `MpcTrajectoryManager` global-sync branch:
  - fixed global replan tick, not per-env dirty age.
  - random sample of `min(num_envs, parallel_plan_batch_size)` envs.
  - `plan_segment(...)` runs only on sampled envs.
  - `reference_reward_mask=True` only for sampled envs with valid MPC results.
  - unsampled envs keep reward mask false.
  - already-valid envs keep following the existing MPC plan between global replan ticks unless reset/command invalidates them.
- Reset/command-change invalidation:
  - `reset_envs(mask)` clears `reference_reward_mask` for those envs.
  - `mark_command_changed(mask)` clears `reference_reward_mask` for those envs.
  - `mark_command_changed(None)` clears all reward mask rows.
- T302g semantic train/play config:
  - removed active `mpc_max_dirty_envs_per_step` setting.
  - added `mpc_replan_mode="global_sync"`.
  - added `mpc_parallel_plan_batch_size=4096`.
- Tests now assert the T302g config no longer depends on dirty-subset budget as the training throughput knob.

## Conclusion

The code now follows the new design at the backend level: `mpc_parallel_plan_batch_size` controls how many envs participate in MPC planning and imitation reward at each global replan tick, and reset/command changes disable the reward even for envs that previously received a valid plan.

## Follow-Up

- Run the real IsaacLab 4096 collect-data timing gate when GPU resources allow.
- Re-run full T302 strict JSONL non-regression before claiming task completion.

## Git Refs

- Baseline Ref: `446a875`
- Candidate Ref: working tree on top of `446a875`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
