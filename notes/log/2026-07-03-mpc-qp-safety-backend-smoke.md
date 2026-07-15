# MPC QP Safety Backend Smoke

## Purpose

Implement and verify the first isolated `mpc_qp` backend scaffold: explicit backend selection, configurable `runtime.qp_iterations`, cache-compatible planner output, viewer/probe selection, sparse semantic/height diagnostics, and a lightweight touchdown semantic keepout repair.

## Stage

MPC-QP backend / trajectory manager runtime / IsaacLab smoke.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

Focused RED/GREEN and regression:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/extension/trajectory_manager_factory.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py Go2Pvcnn/tests/test_mpc_qp_backend.py
git diff --check
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Real IsaacLab smoke on GPU card 1:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_smoke_16.json --planner-backend mpc_qp --qp-iterations 1
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 8 --mpc-num-envs 8 --steps 5 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_smoke_8.json --planner-backend mpc_qp --qp-iterations 1
```

## Input Conditions

- Backend explicitly selected as `planner_backend="mpc_qp"`.
- `runtime.qp_iterations=1`.
- IsaacLab environment: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.
- GPU selection: `CUDA_VISIBLE_DEVICES=1`.
- Current `mpc` backend remains default.

## Key Metrics

Focused tests:

- `Go2Pvcnn/tests/test_mpc_qp_backend.py`: `5 passed`.
- `Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py`: `11 passed`.
- `Go2Pvcnn/tests/test_batch_mpc_backend.py`: `151 passed, 1 warning`.
- `py_compile`: exit `0`.
- `git diff --check`: exit `0`.

Real 16-env smoke:

- exit `0`.
- `planner_backend="mpc_qp"`.
- `qp_iterations=1`.
- `num_envs=16`, `mpc_num_envs=16`, `steps=30`.
- `max_sampled_plan_count_seen=16`.
- `replan_event_count=2`.
- `epoch_seconds=1.210922135040164`.
- CUDA max allocated `106144768` bytes.
- CUDA max reserved `125829120` bytes.

Real 8-env counter smoke:

- exit `0`.
- `planner_backend="mpc_qp"`.
- `runtime_counters.qp_iterations_configured=1`.
- `runtime_counters.qp_iterations_executed=0` on the final non-replan refresh.
- `max_sampled_plan_count_seen=8`.
- CUDA max allocated `67044352` bytes.
- CUDA max reserved `88080384` bytes.

## Result

Pass for first isolated `mpc_qp` scaffold. The backend is opt-in, exports the same reference cache ABI, supports viewer/probe selection, reports QP iteration diagnostics, and passes small real IsaacLab smoke on GPU card 1.

## Conclusion

`mpc_qp` is now wired as an experimental backend without replacing the current `mpc` backend. The current implementation is a safe scaffold using a zero-Adam nominal pass plus sparse diagnostics and touchdown semantic keepout repair; the full constrained QP solver remains the next implementation step behind the new package boundary.

## Follow-Up

- Implement actual batched constrained QP solve instead of the current nominal scaffold.
- Add fixed-shape distance-field constraints and low-small crossing metrics from the design.
- Run 1024/1024 `mpc_qp` performance and memory sweep for `qp_iterations in {1,2,3}` after the real QP core exists.

## Git Refs

- Baseline Ref: `8168b15` (design update for QP iteration tuning).
- Candidate Ref: local working tree.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_qp_planner/](../../Go2Pvcnn/extension/batch_mpc_qp_planner/)
  - [../../Go2Pvcnn/extension/trajectory_manager_factory.py](../../Go2Pvcnn/extension/trajectory_manager_factory.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_mpc_qp_backend.py](../../Go2Pvcnn/tests/test_mpc_qp_backend.py)
  - [../../Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py](../../Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py)
