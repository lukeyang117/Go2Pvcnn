# 2026-07-03 MPC QP Body-Leg Root-Lift Direction Matrix

## Purpose

Close the T302v hard safety gap found by the broader direction matrix: `mixed_turn_l` with `lateral=+0.12` produced FK knee semantic collisions after viewer playback.

## Stage

MPC-QP backend / low-small crossing / FK body-leg semantic safety / GPU1 viewer matrix.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Candidate Ref

Current worktree after adding body-leg XY candidate repair, semantic-volume clearance, and low-small crossing root lift in `mpc_qp`.

## Key Files

- `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
- `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`

## TDD Evidence

RED:

- Added `test_mpc_qp_mixed_turn_repairs_knee_semantic_collision_with_xy_avoidance`.
- The test first failed with `qp_fk_knee_semantic_collision_count=1`.
- After adding the new diagnostics, it also exposed the missing `qp_fk_body_leg_root_lift_count` metric.

GREEN:

- Added fixed-shape body-leg XY candidate repair over foot targets; candidates are scored by recomputed IK/FK knee and shank collision counts.
- Added semantic-volume body-leg clearance so semantic cells have a conservative object-height requirement even when the current scanner height under-reports the future object top.
- Added `low_small_crossing_root_lift_m`: if the horizon crosses a low-small semantic object, the QP applies a conservative root lift over the horizon. A duplicate final-stage lift was removed after GPU1 evidence showed it over-raised the body and worsened IK/readback error.

## Commands And Results

Focused/static:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Result: `15 passed`.

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `172 passed, 1 warning`.

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py

git diff --check
```

Result: both exit `0`.

Real GPU1 hard-case probe:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'mixed_turn_l:0.45,0.15,0.6' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m 0.12
```

Result: exit `0`, `viewer_crossing_acceptance_passed=true`, `fk_semantic_collision_count=0`, collision by part `{"foot":0,"knee":0,"shank":0}`, `fk_touchdown_on_small_rate=0.0`, `fk_foot_small_penetration_rate=0.0`, `root_height_min≈0.400m`, `playback_readback_error_max_m≈0.0594`.

Real GPU1 8-command matrix, `lateral=+0.12`:

- Process exit `0`.
- Hard safety: `max_fk_semantic_collision_count=0`, `max_fk_foot_small_penetration_rate=0.0`, `max_fk_touchdown_on_small_rate=0.0`.
- Strict summary remains `viewer_crossing_acceptance_passed=false` because only `5/8` rows have `fk_foot_over_low_small_success=1`; the remaining rows have no crossing opportunity in that lane rather than a semantic collision.

Real GPU1 8-command matrix, `lateral=-0.12`:

- Process exit `0`.
- Hard safety: `max_fk_semantic_collision_count=0`, `max_fk_foot_small_penetration_rate=0.0`, `max_fk_touchdown_on_small_rate=0.0`.
- Strict summary remains `viewer_crossing_acceptance_passed=false`; `7/8` rows have foot-over success and `diag_fl` still misses the strict lift-then-land/touchdown-after criterion despite zero semantic collision and zero penetration.

Real GPU1 1024/1024 smoke:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_qp_1024_after_body_lift.json --planner-backend mpc_qp --qp-iterations 1
```

Result: exit `0`, `completed_steps=30`, `max_sampled_plan_count_seen=1024`, `max_qp_iterations_executed_seen=1`, `epoch_seconds≈7.146`, CUDA max allocated/reserved `7.51GB/9.27GB`, `max_qp_solve_ms_seen≈25.48`, `max_qp_repair_ms_seen≈10.38`, `max_qp_total_ms_seen≈1665.57`.

## Conclusion

The hard semantic safety gap is closed for the tested two-lane 8-command viewer matrix: FK foot/knee/shank semantic collision, touchdown-on-small, and foot-small penetration are all zero. The previous hard blocker (`mixed_turn_l`, `lateral=+0.12`) now passes.

The full strict foot-over acceptance is still not completely green: the left-lane `diag_fl` row still lacks the strict foot-over/lift-then-land success even though it is collision-free. Treat this as a trajectory-quality/metric follow-up, not as an unresolved semantic-collision blocker.

## Follow-Up

- Decide whether matrix acceptance should count only rows with crossing opportunity, or keep requiring every command row to produce strict foot-over success.
- If strict foot-over must be `8/8` per lane, tune `diag_fl` swing/touchdown-after behavior without relaxing the hard semantic safety checks.
