# T302g MPC Internal Profile Instrumentation

## Purpose

Add printf-style timing inside each MPC planning stage and identify which part of `plan_segment` dominates the 64-env planning cost.

## Stage

`extension/batch_mpc_planner` planner / optimizer / loss registry diagnostics.

## Related Todo

[T302g](../todo/T302g-mpc-semantic-rl-training-config.md), especially `T302g.5a`.

## Commands

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_profile_prints_plan_optimizer_and_loss_stages -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_reference_reward_mask_disables_dirty_unplanned_rows \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_manager_refreshes_only_selected_scanner_rows \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_profile_prints_plan_optimizer_and_loss_stages -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/profiling.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/batch_mpc_planner/optimizer.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py
```

Standalone CUDA profile:

```bash
T302G_MPC_PROFILE_LIMIT=2 T302G_MPC_PROFILE_LOSS_TERMS=20 CUDA_VISIBLE_DEVICES=1 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
...
plan_segment(terrain_64, state_64, command_64, cfg=cfg)
PY
```

4096 IsaacLab rerun attempt:

```bash
T302G_MPC_PROFILE_LIMIT=3 T302G_MPC_PROFILE_LOSS_TERMS=15 \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless --device cuda:1 --num_envs 4096 --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic --planner-backend mpc
```

## Input Conditions

- Device for standalone profile: `CUDA_VISIBLE_DEVICES=1`, process device `cuda:0`.
- Batch: `64`
- Horizon: `25`
- `optimize_steps=24`
- Diagnostics:
  - `T302G_MPC_PROFILE_LIMIT`
  - `T302G_MPC_PROFILE_LOSS_TERMS`
  - `cfg.diagnostics.emit_runtime_counters=True`
  - `cfg.diagnostics.profile_cuda_sync=True` for measured CUDA synchronization.

## Key Metrics

Focused verification:

- profile-output test: `1 passed`
- targeted MPC tests: `3 passed`
- py_compile: exit `0`

Standalone 64-env CUDA profile, stable second run:

- `plan.total_ms=2102.814`
- `optimizer.total_ms=2093.418`
- `optimizer.loop_ms=2057.765`
- `optimizer.loss_ms=945.245`
- `optimizer.backward_ms=1038.696`
- `optimizer.step_ms=22.447`
- `optimizer.grad_clip_ms=24.998`
- `plan.nominal_ms=2.705`
- `plan.postprocess_ms=3.563`

Top forward loss terms in the stable run:

- `touchdown_surface_ms=125.108`
- `ik_fk_residual_ms=85.173`
- `semantic_obstacle_ms=84.774`
- `leg_kinematics_ms=82.605`
- `swing_center_urgency_ms=61.926`
- `obstacle_risk_ms=56.866`
- `swing_direction_ms=54.180`
- `semantic_contact_avoid_ms=49.004`
- `high_obstacle_avoidance_ms=47.465`
- `ik_joint_limit_ms=42.699`

## Result

Pass for instrumentation and standalone MPC profiling. The 4096 IsaacLab rerun did not reach MPC profiling because scene startup hit GPU OOM before environment creation completed.

## Conclusion

The 64-env `plan_segment` cost is dominated by the optimizer loop. Forward loss evaluation and backward are both major contributors. The highest-value next optimization target is not manager cache scatter or semantic scanner refresh, but MPC loss scheduling/query reuse around touchdown surface, semantic obstacle/contact, kinematics, IK/FK, and related terrain sampling. The existing `heavy_loss_stride` and `heavy_loss_enable_from_iter` config fields remain a natural implementation hook because they are already present but not consumed by the loss registry.

## Follow-Up

- Implement quality-preserving heavy-loss scheduling and final full-loss evaluation.
- Preserve strict T302 non-regression by rerunning the JSONL strict metrics after any optimizer/loss scheduling change.
- Retry 4096 headless profiling when GPU memory is available; the latest failure reported GPU1 had only about `533MB` free before ray-caster allocation.

## Git Refs

- Baseline Ref: working tree on top of `946811f`
- Candidate Ref: working tree, 2026-05-18 13:38 CST
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/profiling.py](../../Go2Pvcnn/extension/batch_mpc_planner/profiling.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py](../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
