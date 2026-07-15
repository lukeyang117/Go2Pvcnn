# 2026-07-03 MPC QP Safety Core And Sweep

## Purpose

Verify the second `mpc_qp` implementation pass: fixed-shape safety QP core, low-small crossing diagnostics, configurable QP iterations, and GPU1 IsaacLab smoke/sweep.

## Stage

MPC-QP backend / trajectory manager runtime / IsaacLab high-concurrency probe.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Candidate Ref

Current worktree after adding:

- `Go2Pvcnn/extension/batch_mpc_qp_planner/distance_field.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- QP diagnostics in `constraints.py`
- QP integration in `planner.py`
- perf-probe historical QP counters

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_core_smoke_16_v2.json --planner-backend mpc_qp --qp-iterations 1
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_core_smoke_16_qp2.json --planner-backend mpc_qp --qp-iterations 2
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_core_smoke_16_qp3.json --planner-backend mpc_qp --qp-iterations 3
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_core_smoke_1024_qp1.json --planner-backend mpc_qp --qp-iterations 1
```

## Results

- Focused QP tests: `10 passed`.
- QP + participation + current MPC backend regression: `167 passed, 1 warning`.
- Pycompile: exit `0`.
- 16/16, `qp_iterations=1`: exit `0`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `qp_replan_event_count=2`, CUDA allocated/reserved `106372096/125829120`.
- 16/16, `qp_iterations=2`: exit `0`, `max_qp_iterations_executed_seen=2`, CUDA allocated/reserved `113499136/127926272`.
- 16/16, `qp_iterations=3`: exit `0`, `max_qp_iterations_executed_seen=3`, CUDA allocated/reserved `107866112/127926272`.
- 1024/1024, `qp_iterations=1`: exit `0`, `max_sampled_plan_count_seen=1024`, `max_qp_iterations_executed_seen=1`, epoch seconds `7.4591`, CUDA allocated/reserved `7442613248/9225371648`.

## Design Metrics Covered

- `touchdown_on_small_count == 0` covered by focused low-small test.
- `crossing_leg_count > 0` covered by focused low-small test.
- `fk_semantic_collision_count == 0`, `fk_semantic_collision_rate == 0`, and non-negative foot-path clearance over small semantic cells covered for foot path samples.
- `max_semantic_constraint_violation <= tolerance`, `max_height_constraint_violation <= tolerance`, and `step_cap_violation_count == 0` covered by focused QP tests.
- `terrain_risk_reduces_target_progress == true` covered by high height-variation test.
- 1024/1024 memory stayed within the accepted T302u envelope of about `7.5GB/9.3GB`.

## Remaining Gaps

- Full FK knee/shank/root-underbody constrained QP sampling is not complete; current pass covers touchdown and foot-path safety first.
- Viewer visual acceptance for `go2_foostep_planner.py --planner-backend mpc_qp` was not rerun after this QP core pass.
- Dedicated QP solve timing is not separated from nominal parametric export/runtime profile yet.

## Conclusion

Pass for the current QP core milestone. Do not mark full T302v complete until FK body/leg constraints and viewer acceptance are closed.
