# T302l PLAY / VIEWER Cfg Split Verification

## Purpose

Verify that ordinary policy `play.py` uses a no-MPC PLAY config, while the MPC viewer keeps the MPC-enabled VIEWER config and the low-small MPC hard metrics do not regress.

## Stage

MPC semantic RL runtime / play-viewer entrypoint split.

## Related Todo

- [T302l](../todo/T302l-mpc-rl-participation-and-reward-plan.md)
- [Task 20: PLAY / VIEWER Cfg Split](../todo/T302l-mpc-rl-participation-and-reward-plan.md#task-20-play--viewer-cfg-split)

## Git Refs

- Baseline Ref: `d6f77d7`
- Candidate Ref: current working tree
- Current Work Ref: `costmap-teacher-ablation`

## Key Files

- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
- [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
- [../../Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic_cnn.py](../../Go2Pvcnn/rsl_rl/rsl_rl/modules/actor_critic_cnn.py)
- [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)

## Commands

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "mpc_semantic or teacher_mpc_semantic_env_raises_fk_body_leg_collision_weight or cleanup_entrypoints"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/play.py --headless --device cuda:0 --num_envs 1 --experiment teacher_elevation_trajectory_mpc_semantic --run_dir 2026-05-31_20-03-27 --checkpoint model_14000.pt --max-steps 5 --debug-livestream
CUDA_VISIBLE_DEVICES=0 timeout 300s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants parametric_v1 --cycles 1 --requested-n-frames 300
```

## Input Conditions

- Conda env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Card 3 avoided as requested.
- PLAY policy checkpoint:
  `logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-05-31_20-03-27/model_14000.pt`

## Key Metrics

- Static/unit regression:
  - `test_viewer_reset.py + test_mpc_semantic_rl_env_cfg.py`: `28 passed`
  - focused `test_batch_mpc_backend.py`: `11 passed, 118 deselected, 1 warning`
- PLAY smoke:
  - exit code `0`
  - `[Policy] Loaded successfully`
  - actor input dim `1069`, critic input dim `1072`
  - `Starting Play Loop`
  - `Play Complete - Timesteps: 5`
  - no `[Planner] Attached ... trajectory manager` line
- low-small regression:
  - rows: `5`
  - crossing-covered rows: `2`
  - max covered `fk_semantic_collision_count`: `0`
  - max covered `fk_semantic_collision_rate`: `0.0`
  - max covered `planned_vs_fk_foot_error_crossing_leg_max_m`: `0.04160968214273453`

## Result

Pass.

## Conclusion

`TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY` is suitable for ordinary headless policy playback without attaching MPC. `TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER` remains the MPC viewer route, and the low-small hard gate remains inside the accepted envelope: no FK semantic collision on covered crossing rows and crossing-leg FK error below `0.08m`.

## Follow-up

No new loss or hard constraint was added. Existing unrelated dirty files and deleted `${data}` / old docs artifacts were not touched.
