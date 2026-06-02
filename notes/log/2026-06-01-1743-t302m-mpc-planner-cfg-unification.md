# T302m MPC planner cfg unification

- Purpose: unify RL task-side MPC tuning under the real `MpcPlannerCfg` object.
- Stage: teacher elevation MPC semantic cleanup / MPC tuning contract.
- Related todo: [T302m](../todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md)

## Command / Procedure

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

## Input Conditions

- Local working tree only.
- Production task cfg now carries `mpc_planner_cfg: MpcPlannerCfg`.
- Legacy top-level MPC aliases removed from the active task config.

## Key Metrics

- `test_viewer_reset.py`: `15 passed`
- `test_mpc_rl_participation.py` + `test_batch_mpc_backend.py`: `133 passed, 1 warning`

## Result

Pass. The task config now edits planner runtime/diagnostics/participation through the real MPC config object instead of duplicated RL-side alias fields.

## Conclusion

The MPC tuning surface is now unified around one config object. Old alias-based task overrides remain only as compatibility fallback for legacy fake configs/tests.

## Follow-up

- Keep `planner_cfg_from_task_cfg()` compatibility fallback only if legacy consumers still need it.
- If the user wants full cleanup later, remove the remaining alias bridge and update all fake configs to pass `mpc_planner_cfg` directly.

## Git Refs

- Baseline Ref: `d6f77d7`
- Candidate Ref: working tree at 2026-06-01 17:43
- Key Files:
  - [Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
