# T302 Expanded MPC Headless Metrics

## Purpose

Broaden T302 metric coverage after the implementation pass, following the user request to debug and test small-obstacle collision rate, low-small crossing, high-small/large avoidance, complex-terrain collision rate, and root/knee/shank collision rates in the real IsaacLab headless environment.

## Stage

Real IsaacLab headless acceptance for the active `Go2Pvcnn/extension/batch_mpc_planner` MPC backend.

## Related Todo

- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Baseline Ref

- `769f7d4`

## Candidate Ref

- Working tree on top of `769f7d4`

## Key Files

- [../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py](../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py)
- [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)

## Commands

```bash
MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -q

MPC_T302_HEADLESS=1 MPC_TEST_DEVICE=cuda:0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -vv --tb=short

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py \
  Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/semantic_course.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py

git diff --check
```

## Input Conditions

- Python env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Device: `cuda:0`
- `MPC_T302_HEADLESS=1`
- Planner backend: `mpc`
- New fixture command cases:
  - `forward_yaw_left`
  - `forward_yaw_right`
  - `diagonal_forward_left`
  - `diagonal_forward_right`
- Semantic-small height override cases:
  - low small: `0.16m`
  - high small: `0.46m`

## Key Metrics

- Backend MPC suite: `51 passed`
- Expanded T302 real headless suite:
  - collected `6` tests in verbose run
  - command exit code `0`
- Py compile: exit code `0`
- Diff whitespace check: exit code `0`
- Expanded headless assertions now cover:
  - COBBLESTONE commands: forward, backward, lateral left/right, yaw left/right, forward+yaw left/right, diagonal forward left/right.
  - root-bottom collision ratio on COBBLESTONE.
  - swing-foot collision ratio on COBBLESTONE and semantic obstacles.
  - knee and shank height-field collision ratios.
  - low-small crossing along forward/back/lateral command directions.
  - low-small stance semantic obstacle count and ratio equal zero.
  - high-small command-direction tracking risk scale and non-crossing/clearance behavior.
  - large forward obstacle risk scale and root clearance from large obstacle footprint.
  - large yaw obstacle risk scale.

## Result

Pass.

## Conclusion

No new production MPC code change was required in this follow-up because the broadened real headless metric suite passed. The change strengthens the acceptance tests so future MPC edits must preserve small-obstacle crossing, high-small/large avoidance/deweight behavior, and root/knee/shank/swing-foot height-field collision safety across more command combinations.

## Follow-Up

- The remaining risk is longer unrolled command-switch/yaw playback and large-scale 4096 runtime counter extraction if T302 is promoted into a full RL rollout target.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: working tree on top of `769f7d4`
