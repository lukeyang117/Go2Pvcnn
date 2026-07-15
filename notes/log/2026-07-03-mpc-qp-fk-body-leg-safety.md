# 2026-07-03 MPC QP FK Body Leg Safety

## Purpose

Close the next T302v gap: `mpc_qp` must report and satisfy FK knee/shank/root-underbody semantic and height safety metrics, not only touchdown and foot-path metrics.

## Stage

MPC-QP backend / projected safety QP / FK body-leg collision diagnostics.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Candidate Ref

Current worktree after adding FK/body diagnostics and QP shank clearance lift.

## Key Files

- `Go2Pvcnn/extension/batch_mpc_qp_planner/constraints.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`

## TDD Evidence

RED:

- `test_mpc_qp_reports_fk_leg_and_underbody_collision_free_metrics` first failed with missing `qp_fk_body_leg_collision_count`.
- After adding diagnostics, the same test exposed real unsafe FK output: `qp_fk_body_leg_collision_count=2`, `qp_fk_shank_semantic_collision_count=2`, and `qp_fk_body_leg_height_violation_max≈0.0818m`.

GREEN:

- QP foot corrections now recompute `joint_angles`.
- QP step adds fixed-shape FK shank clearance lift for swing frames and recomputes IK.
- Sparse diagnostics now cover knee, shank, underbody semantic counters and FK/body height violation max.

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_reports_fk_leg_and_underbody_collision_free_metrics -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
git diff --check
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 16 --mpc-num-envs 16 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_fk_body_smoke_16.json --planner-backend mpc_qp --qp-iterations 1
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_fk_body_smoke_1024.json --planner-backend mpc_qp --qp-iterations 1
```

## Results

- Focused FK/body test: pass.
- Full QP suite: `11 passed`.
- QP + participation + current MPC backend regression: `168 passed, 1 warning`.
- Pycompile: exit `0`.
- Diff check: exit `0`.
- 16/16 GPU1 smoke: exit `0`, `max_sampled_plan_count_seen=16`, `max_qp_iterations_executed_seen=1`, CUDA allocated/reserved `114159104/130023424`.
- 1024/1024 GPU1 smoke: exit `0`, `max_sampled_plan_count_seen=1024`, `max_qp_iterations_executed_seen=1`, epoch seconds `7.4477`, CUDA allocated/reserved `7402625024/9227468800`.

## Design Metrics Covered

- `root_underbody_collision_count == 0`
- `fk_body_leg_collision_count == 0`
- FK knee semantic collision count `0`
- FK shank semantic collision count `0`
- underbody semantic collision count `0`
- FK/body height violation max `<= 1e-5` in the focused low-small crossing test

## Remaining Gaps

- Viewer visual acceptance after FK/body QP changes is still pending.
- Dedicated QP solve/linearization timing fields are still not separated from the nominal planner profile.
- Real low-small visual/gameplay proof still needs viewer or controlled trajectory playback evidence beyond unit-level planned-reference metrics.
