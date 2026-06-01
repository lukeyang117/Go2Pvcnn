# 2026-05-31 23:06 T302m Card1 IsaacLab Acceptance

## Purpose

Re-run T302m real IsaacLab acceptance on card1 after card3 was blocked by an existing 1024-env long training process.

## Stage

IsaacLab runtime acceptance for single-route `teacher_elevation_trajectory_mpc_semantic + mpc` cleanup.

## Related Todo

- [../todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md](../todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md)

## Environment

- Conda/Python: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- GPU selection: `CUDA_VISIBLE_DEVICES=1`
- Card1 was effectively free before testing (`27MiB / 24564MiB`, only Xorg listed).

## Commands And Results

### Semantic Contact Robot Drop Probe

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_semantic_contact_robot_drop_probe.py::test_semantic_contact_robot_drop_probe_real_isaaclab_small -q
```

Result: pass, exit code `0`.

### 1024 Env Train Smoke

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless --device cuda:0 --num_envs 1024 --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic --planner-backend mpc
```

Result: pass, exit code `0`.

Key runtime evidence:

- Registered only semantic MPC Gym ids.
- Created `1024` envs.
- Attached `mpc` trajectory manager.
- Reward manager contained `reference_foot_pos` and `semantic_contact_collision`.
- `ActorCriticCNN` initialized with actor input `301`, critic input `304`, actions `12`.
- First five step timing lines were emitted.
- `reference_foot_pos` first call was about `2187ms`, then about `2ms` on subsequent sampled steps.
- `semantic_contact_collision` was about `1.1-1.5ms` on sampled steps.

Fixes required before pass:

- `train.py`/`play.py` now add local `Go2Pvcnn/rsl_rl` to `sys.path` and import `rsl_rl`, not the stale `rsl_rl_2_01` alias.
- `SimpleRslRlEnvWrapper` now returns `(policy_obs, extras)` where `extras["observations"]["critic"]` contains critic observations, matching local RSL-RL `VecEnv` / `OnPolicyRunner`.
- Removed unsupported `rnd_cfg` and `symmetry_cfg` from the active `agent/train_cfg.py` PPO config.

### 1024 Env / 64 MPC Env / 25 Step Perf Probe

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
```

Result: pass, exit code `0`.

Output JSON:

```json
{"num_envs": 1024, "selected_mpc_envs": 64, "epoch_seconds": 5.882832678034902}
```

Acceptance target: `epoch_seconds < 10s`; observed `5.8828s`.

### Local Regression After Runtime Fixes

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py \
  Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_semantic_contact_rewards.py \
  Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py Go2Pvcnn/tests/test_viewer_reset.py \
  Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py -q
```

Result: `171 passed, 1 warning in 4.33s`.

Additional checks:

- `python -m py_compile Go2Pvcnn/scripts/train.py Go2Pvcnn/scripts/play.py Go2Pvcnn/agent/train_cfg.py`: pass.
- Production old-route scan for `rsl_rl_2_01`, old planner backends, old teacher cfg imports, and debug variants: no matches.

## Conclusion

Card1 acceptance passes for the cleaned single-route code path. The earlier card3 failure was a resource conflict, not the final code state.

## Git Refs

- Current Work Ref: local working tree on branch `costmap-teacher-ablation`.
- Key Files: `Go2Pvcnn/scripts/train.py`, `Go2Pvcnn/scripts/play.py`, `Go2Pvcnn/agent/train_cfg.py`, `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`, `Go2Pvcnn/extension/trajectory_manager_factory.py`, `Go2Pvcnn/extension/batch_mpc_planner/`, `Go2Pvcnn/tests/`.
