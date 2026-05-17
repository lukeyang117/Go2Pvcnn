# T302 MPC Body/Leg Collision Implementation Verification

## Purpose

Verify the T302 MPC implementation for body/leg height-field collision safety, semantic stance/touchdown obstacle rejection, high-obstacle tracking risk scaling, and compact real IsaacLab headless acceptance.

## Stage

Production `Go2Pvcnn/extension/batch_mpc_planner` MPC backend plus viewer/runtime fixtures used for headless acceptance.

## Related Todo

- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Baseline Ref

- `769f7d4`

## Candidate Ref

- Working tree on top of `769f7d4`

## Key Files

- [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
- [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
- [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
- [../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py](../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py)

## Command

```bash
MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/semantic_course.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py
```

## Input Conditions

- Python env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Headless fixture device: `cuda:0`
- T302 headless enabled by `MPC_T302_HEADLESS=1`
- New headless cases:
  - COBBLESTONE terrain planned root/body/foot/knee/shank clearance.
  - Low semantic-small obstacle crossing without stance on obstacle ids.
  - Large obstacle yaw-risk scaling and stance semantic rejection.

## Key Metrics

- Backend MPC suite: `51 passed`
- T302 headless suite: `3 passed`
- Py compile: exit code `0`
- T302 backend contracts covered:
  - knee/shank FK output shapes.
  - body and leg height-field collision losses.
  - stance semantic obstacle loss using semantic ids only.
  - all-scanner obstacle risk scaling for linear and yaw-only commands.
  - `cost_breakdown` and `loss_breakdown` expose collision/risk diagnostics.
  - GPU hot-path guardrail rejects `.cpu().numpy()` and per-env loop patterns in T302 loss hot paths.

## Result

Pass.

## Conclusion

T302 is implemented in the active MPC backend. The implementation preserves T300e no-memory/no-output-grounding contracts while adding collision/risk semantics and compact real IsaacLab headless acceptance.

## Follow-Up

- Broaden beyond compact acceptance with longer command-switch/yaw sequences and 4096-scale counters if this branch becomes the active RL training rollout target.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: working tree on top of `769f7d4`
