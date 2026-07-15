# 2026-07-03 MPC QP Stage Timing Counters

## Purpose

Close the T302v observability gap for `mpc_qp`: split nominal planning, QP solve, repair, diagnostics, and total backend time so speed/memory tuning is not inferred only from whole-epoch seconds.

## Stage

MPC-QP backend / runtime counters / IsaacLab GPU1 performance probe.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Candidate Ref

Current worktree after adding QP timing fields to planner diagnostics, manager runtime counters, and perf probe historical summaries.

## Key Files

- `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- `Go2Pvcnn/extension/batch_mpc_planner/manager.py`
- `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`
- `Go2Pvcnn/tests/test_mpc_qp_backend.py`

## TDD Evidence

RED:

- `test_mpc_qp_perf_probe_records_historical_qp_iteration_metrics` failed because `max_qp_solve_ms_seen` was missing.
- `test_mpc_qp_plan_segment_reports_stage_timing_diagnostics` failed because `qp_nominal_ms` and sibling timing fields were missing from `result.loss_breakdown`.

GREEN:

- `plan_segment_qp()` now reports `qp_nominal_ms`, `qp_solve_ms`, `qp_repair_ms`, `qp_diagnostics_ms`, and `qp_total_ms`.
- `MpcTrajectoryManager` extracts finite max/mean QP timing values from result diagnostics into `runtime_counters`.
- `mpc_rl_epoch_perf_probe.py` records historical maxima: `max_qp_nominal_ms_seen`, `max_qp_solve_ms_seen`, `max_qp_repair_ms_seen`, `max_qp_diagnostics_ms_seen`, and `max_qp_total_ms_seen`.

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
git diff --check
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_timing_smoke_16.json --planner-backend mpc_qp --qp-iterations 1
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_timing_smoke_1024.json --planner-backend mpc_qp --qp-iterations 1
```

## Results

- Focused QP suite: `12 passed`.
- QP + participation + current MPC backend regression: `169 passed, 1 warning`.
- Pycompile: exit `0`.
- Diff check: exit `0`.
- 16/16 GPU1 smoke: exit `0`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `epoch_seconds≈1.30`, `max_qp_solve_ms_seen≈7.01`, `max_qp_total_ms_seen≈186.82`, CUDA allocated/reserved `108250624/130023424`.
- 1024/1024 GPU1 smoke: exit `0`, `max_sampled_plan_count_seen=1024`, `max_qp_iterations_executed_seen=1`, `epoch_seconds≈7.28`, `max_qp_solve_ms_seen≈12.44`, `max_qp_total_ms_seen≈1668.19`, CUDA allocated/reserved `7573121024/9279897600`.

## Conclusion

The QP-specific solve time is now separately observable from the nominal zero-Adam trajectory export cost. High-concurrency memory remains in the previously observed envelope, and `mpc_qp` still requires explicit backend selection.

## Follow-up

Viewer or controlled playback evidence remains pending for post-core low-small crossing without contact.
