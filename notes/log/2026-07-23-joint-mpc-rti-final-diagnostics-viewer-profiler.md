# Final RTI Diagnostics, Viewer, And Profiler

## Purpose

Complete final-plan Task 13 without adding a second selector, LQ/QP, line search, or SQP call.

## Stage And Refs

- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Baseline Ref: `878f5ab`
- Candidate Ref: current Task 13 checkpoint
- Branch: `work/joint-mpc-kinematic`

## Implementation

- Added eight ordered profiler stages using preallocated CUDA events, with synchronization deferred to the external viewer/benchmark caller.
- Exposed the manager's complete latest `JointMpcRtiStepResult`; the viewer consumes trajectory and diagnostics from the same refresh.
- Carried full nominal/direction, 25 touchdown candidates and reject masks, selected/previous targets, convex-region geometry, five alpha candidates, gait phase, publish/stop, KKT, slack, active rows, and clearance diagnostics.
- Added viewer markers for safe/rejected candidates, previous target, region corners, nominal root, five alpha root paths, selected target, and final trajectory.
- Added optional `--joint-mpc-profile`; normal CUDA Graph execution is unchanged.

## Verification

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_diagnostics.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rti_pipeline.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py \
  Go2Pvcnn/tests/test_viewer_reset.py -q
```

Result: `61 passed in 24.05s`, including real CUDA profiler events and CUDA Graph rolling capture/replay.

## Conclusion

Task 13's fixed diagnostics, profiler, and same-refresh viewer path pass. Real marker appearance remains part of later canonical flat/small/large evidence; behavior and performance are not claimed here.

## Key Files

- `Go2Pvcnn/extension/joint_mpc_rti/diagnostics/profiler.py`
- `Go2Pvcnn/extension/joint_mpc_rti/planner.py`
- `Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py`
- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- `Go2Pvcnn/tests/test_viewer_reset.py`
