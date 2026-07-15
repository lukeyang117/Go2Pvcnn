# 2026-07-07 MPC QP Zero-Command Hold Last Plan Frame

## Purpose

Implement the user's corrected idle behavior: when velocity command is zero, `mpc_qp` should not replan from the current physical state. It should use the previous planned reference's final frame as the new static reference.

## Stage

MPC-QP trajectory manager / reference cache behavior.

## Related Todo

[T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Change

- Added a small manager hook in the base `MpcTrajectoryManager`; default `mpc` behavior returns all selected rows for normal planning.
- Overrode the hook in `MpcQpTrajectoryManager`.
- For selected `mpc_qp` rows with `command_norm <= runtime.idle_command_threshold` and a valid previous cache:
  - remove those rows from the QP planning batch
  - copy the previous cache's final frame across the full horizon for root, quat, joints, feet, contact, and touchdown fields
  - reset those rows' phase counter to `0`
- Nonzero command rows still go through normal `mpc_qp` planning.

This is a cache-level hold, not a QP loss change and not a hard repair/candidate endpoint change.

## Verification

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_manager_holds_previous_plan_final_frame_for_zero_command_rows -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/batch_mpc_qp_planner/manager.py Go2Pvcnn/tests/test_mpc_qp_backend.py
git diff --check -- Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/batch_mpc_qp_planner/manager.py Go2Pvcnn/tests/test_mpc_qp_backend.py notes/todo.md notes/log/index.md notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md
```

Results:

- focused manager test: `1 passed`
- full QP unit suite: `67 passed`
- base MPC suite: `153 passed, 1 warning`
- pycompile: pass
- diff check: pass

Additional direct playback probe:

```bash
CUDA_VISIBLE_DEVICES=3 timeout 180s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py --device cuda:0 --planner-backend mpc_qp --terrain task --cycles 4 --pre-command 0.45,0,0 --pre-cycles 2 --requested-n-frames 25 --playback-frames 25 --qp-iterations 1
```

The probe's idle cycles report root/foot/joint step `0`. This probe calls viewer planning directly, so it does not exercise the manager cache-hold path; the manager behavior is covered by the focused unit test.

## Conclusion

`mpc_qp` now has the requested zero-command manager behavior: use the previous plan's final frame as the new static reference and skip QP planning for those rows.

## Git Refs

- Baseline Ref: dirty worktree after idle jitter A/B diagnostics.
- Candidate Ref: dirty worktree after manager-level zero-command hold.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/manager.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
