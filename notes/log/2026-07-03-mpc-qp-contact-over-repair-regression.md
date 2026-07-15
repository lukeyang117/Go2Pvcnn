# 2026-07-03 MPC QP Contact-Over Repair Regression

## Purpose

Record the follow-up fix for the `mpc_qp` low-small contact-over repair diagnostic and verify that the isolated QP backend still passes static regression plus real IsaacLab GPU1 smoke after the fix.

## Stage

MPC-QP backend / low-small semantic crossing / fixed-shape repair diagnostics / GPU1 IsaacLab smoke.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Baseline Ref

`8168b15` plus existing dirty T302v worktree.

## Candidate Ref

Current dirty worktree after adding `qp_low_small_contact_over_repair_count` and contact-foot height repair in `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`.

## Key Files

- `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`

## TDD Evidence

RED:

- `pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q`
- Result before fix: `1 failed, 16 passed`
- Failure: `test_mpc_qp_lifts_contact_leg_when_low_small_lies_on_crossing_path` raised `KeyError: 'qp_low_small_contact_over_repair_count'`.

GREEN:

- Added a fixed-shape low-small contact-over repair path. If `semantic == 1` is under a contact foot sample, the QP backend lifts that foot sample to `terrain_z + low_small_swing_clearance_m`, recomputes IK, and reports `qp_low_small_contact_over_repair_count`.
- Added a default zero diagnostic in `plan_segment_qp()` so rows without contact-over repair keep a stable metric key.

## Commands And Results

Focused failed case:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_lifts_contact_leg_when_low_small_lies_on_crossing_path -q
```

Result: `1 passed`.

Focused QP suite:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Result: `17 passed`.

Current MPC/QP regression:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `174 passed, 1 warning`.

Static checks:

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py
git diff --check
```

Result: both exit `0`.

GPU availability before real smoke:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Relevant result: card `1` was an `NVIDIA GeForce RTX 4090` with `27 MiB / 24564 MiB` used and `0 %` utilization before the smoke.

Real 16/16 GPU1 smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_contact_repair_smoke_16.json --planner-backend mpc_qp --qp-iterations 1
```

Result: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, `epoch_seconds=1.263`, `max_qp_solve_ms_seen=13.709`, `max_qp_repair_ms_seen=9.354`, CUDA allocated/reserved `113851392 / 130023424` bytes.

Real 1024/1024 GPU1 smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_contact_repair_smoke_1024.json --planner-backend mpc_qp --qp-iterations 1
```

Result: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=1024`, `max_qp_iterations_executed_seen=1`, `epoch_seconds=7.671`, `max_qp_nominal_ms_seen=1628.211`, `max_qp_solve_ms_seen=149.440`, `max_qp_repair_ms_seen=139.939`, `max_qp_total_ms_seen=1926.422`, CUDA allocated/reserved `7755427840 / 9491709952` bytes.

## Conclusion

The missing contact-over diagnostic was a real implementation gap, now fixed. The `mpc_qp` backend still passes the focused QP suite, current MPC compatibility regression, pycompile/diff checks, and real GPU1 16/16 plus 1024/1024 IsaacLab smokes.

The 1024/1024 memory stays inside the accepted 24GB-card envelope and close to previous T302v memory evidence. QP solve and repair timing are higher than the previous body-leg matrix log, so future speed work should profile the new contact-over and body-leg repair passes before adding denser constraints.

## Follow-Up

- Keep `qp_iterations=1` as the default.
- If strict matrix acceptance must become `8/8`, tune the remaining `diag_fl` strict lift-then-land/touchdown-after behavior without relaxing hard semantic safety.
