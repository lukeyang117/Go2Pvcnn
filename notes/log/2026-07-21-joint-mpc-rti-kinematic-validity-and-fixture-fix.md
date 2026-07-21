# Joint MPC RTI Validity And Flat Fixture Fix

## Purpose

Close two Task 13 diagnostic defects before another ranked CUDA flat run: the acceptance fixture used a root height inconsistent with the configured flat contact geometry, and planner validity was inherited from cold nominal reachability even when line search accepted a finite non-nominal candidate.

## Stage

Task 13 flat behavior diagnosis. This is regression evidence only; it is not a flat acceptance result.

## Procedure

- Added a RED geometry regression showing that the shared initial state must use `posture_root_clearance` and keep the default FK feet at or above `foot_contact_offset`.
- Added a RED planner regression that forces `nominal.valid=False` while returning an accepted finite non-nominal RTI state.
- Changed the fixture root height from the stale literal `0.32m` to the configured `0.34m` posture clearance.
- Changed final validity to require a valid nominal only for alpha-zero fallback; an accepted non-nominal candidate instead requires finite state and solver status zero.
- Ran:

  `PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py Go2Pvcnn/tests/joint_mpc_rti/test_joint_metrics.py Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py -q`

## Key Metrics

- Focused regression: `50 passed in 10.92s`.
- Default fixture root Z: `posture_root_clearance = 0.34m`.
- Final validity still requires finite state and `update.status == 0`.
- Nominal fallback still requires `nominal.valid`; no line-search filter, loss, nominal construction, KKT constraint, or recovery path was added.

## Result

Pass for the two scoped regressions. The prior mixed-command ranked behavior metrics remain unverified after this change, so Task 13 and the flat gate remain open.

## Follow-up

Run the same 3-cell, 24-step flat matrix under the monitored CUDA runner and inspect accepted alpha, per-step validity, joint filters, foot/root onset, clearance, and zero drift before tuning any approved parameter.

## Git Refs

- Baseline ref: `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`, `Go2Pvcnn/tests/joint_mpc_rti/helpers.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_flat_acceptance.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py`
