# T302g MPC Semantic RL Implementation

## Purpose

Implement the first T302g code slice: independent MPC semantic train/play task, semantic CNN map, swing-leg collision reward, dirty-subset/current-state regression coverage, and a 4096 headless timing gate entry.

## Stage

Independent MPC semantic RL config under `Go2Pvcnn/go2_pvcnn/tasks/`, planner manager attachment, MDP observations/rewards, train/play entrypoints, and headless runtime tests.

## Related Todo

- [../todo/T302g-mpc-semantic-rl-training-config.md](../todo/T302g-mpc-semantic-rl-training-config.md)

## Commands

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  -k "mpc_semantic or downsampled_semantic or swing_leg_collision or dirty or current" -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q

MPC_RUNTIME_4096=1 MPC_TEST_DEVICE=cuda:1 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_semantic_runtime_4096_collect_data_under_10s -q -s

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/extension/mdp/observations.py \
  Go2Pvcnn/extension/mdp/rewards_reference.py \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py
```

## Input Conditions

- Python: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- 4096 runtime command used `MPC_TEST_DEVICE=cuda:1` and headless AppLauncher fixture.
- Existing `teacher_elevation_trajectory` default remains `planner_backend="together"`; new T302g task explicitly defaults to `planner_backend="mpc"`.

## Key Metrics

- T302g-focused backend slice: `12 passed, 67 deselected`.
- Full backend regression file: `79 passed`.
- `py_compile`: exit `0`.
- 4096 headless timing-gate pytest: exit `0`, but current Isaac AppLauncher/pytest output did not flush the test summary, metric print, or metrics JSON; therefore the `collect_time_s` numeric value is not accepted as recorded evidence yet.

## Result

Partial pass.

- Implemented new config file `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`.
- Registered new Gym ids:
  - `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0`
  - `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0`
- Added train/play experiment choice `teacher_elevation_trajectory_mpc_semantic`; play now accepts `--planner-backend mpc`.
- Added semantic CNN helper returning `2 x 16 x 16` maps from a high-resolution semantic scanner, with area-pooled elevation and priority-pooled semantic ids.
- Added `swing_leg_collision_reward` from current IsaacLab body/contact/scanner buffers and wired it into the new task.
- Extended trajectory-manager attach allowlist to include the new experiment.
- Extended 4096 runtime fixture/test path to target the new task instead of the old viewer task.

## Conclusion

T302g.1, T302g.2, T302g.3, and the already-existing T302g.4 manager invariants now have focused test coverage. T302g.5 has a test gate entry but still needs a trustworthy recorded `collect_time_s` value from a runtime path that does not lose pytest/test-body output on SimulationApp shutdown. T302g.6 has backend non-regression evidence but not the full T302 strict JSONL rerun.

## Follow-up

- Capture 4096 collect-data timing with a harness that writes metrics before SimulationApp shutdown or avoids the current pytest/AppLauncher output loss.
- Run the T302 strict JSONL probe path from [2026-05-17-0804-t302-strict-collision-metric-tuning.md](2026-05-17-0804-t302-strict-collision-metric-tuning.md).

## Git Refs

- Baseline Ref: `946811f`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/extension/mdp/observations.py](../../Go2Pvcnn/extension/mdp/observations.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
