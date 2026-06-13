# 2026-06-13 20:22 Play Disable Timeout Refresh

## Purpose

按用户要求删除 PLAY 配置里的 timeout 自动刷新，让 policy 可视化和 livestream 手动控制时不再因为 episode 到时而周期性 reset。

## Stage

Play cfg / visualization lifecycle.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

RED:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py::test_play_cfgs_disable_timeout_refresh_for_visualization -q
```

Observed before implementation:

```text
1 failed
```

Focused verification:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py::test_play_cfgs_disable_timeout_refresh_for_visualization \
  Go2Pvcnn/tests/test_viewer_reset.py::test_flat_small_play_cfg_disables_training_curriculum_without_semantic_contact_sensors \
  -q
python -m py_compile Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Broader static verification:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play \
  -q
```

Real smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/play.py \
  --headless --device cuda:0 --num_envs 1 --max-steps 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --run_dir 2026-06-12_19-05-27 --checkpoint model_28900.pt \
  --terrain-row 3 --terrain-col 0
```

## Input Conditions

- Flat-small PLAY cfg.
- Checkpoint: `2026-06-12_19-05-27/model_28900.pt`.
- Keyboard control not needed for this smoke.

## Key Evidence

- Focused tests:

```text
2 passed
```

- Full viewer reset:

```text
33 passed
```

- Backend cfg subset:

```text
2 passed
```

- Pycompile exits `0`.
- Real smoke exits `0`.
- Real smoke Termination Manager:

```text
[INFO] Termination Manager:  <TerminationManager> contains 2 active terms.
|   0   | base_contact    |  False   |
|   1   | bad_orientation |  False   |
```

## Result

Pass. `TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY` and `TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY` set `self.terminations.time_out = None`.

## Conclusion

PLAY visualization no longer has timeout-based automatic episode refresh. Training cfg still keeps `time_out`, so curriculum success logic is not changed.

## Follow-Up

The real smoke printed many Isaac/Carb `Failed to create change watch ... errno=28/No space left on device` messages before continuing. This looks like an environment watch/inotify resource warning, not a failure of the timeout change, because the smoke exited `0`.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
