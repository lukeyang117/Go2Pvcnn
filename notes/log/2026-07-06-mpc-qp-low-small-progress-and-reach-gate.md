# 2026-07-06 MPC QP Low-Small Progress And Reach Gate

## Purpose

Use the old MPC low-small probe metrics as a reference for the current continuous `mpc_qp` path, after the user pointed to the existing `Go2Pvcnn/tests` small-obstacle probes and metrics.

## Stage

MPC-QP backend / continuous Bezier trajectory / low-small obstacle crossing.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

Static:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py
```

Small obstacle:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py \
  --device cuda:0 \
  --commands 'forward:0.45,0.0,0.0' \
  --cycles 1 \
  --requested-n-frames 50 \
  --playback-frames 50 \
  --qp-iterations 2
```

Hard terrain:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py \
  --device cuda:0 \
  --tiles 8:12 \
  --commands 'forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0' \
  --cycles 1 \
  --requested-n-frames 25 \
  --playback-frames 25 \
  --qp-iterations 2 \
  --warmup-steps 4
```

## Changes

- Added a continuous low-small root progress residual. It uses fixed command-frame samples and semantic/height field queries to detect a low-small obstacle in the forward corridor, then moves the root trajectory and foot controls together far enough to create a crossing window.
- Added a continuous low-small foot-over residual with a reachability gate. If lateral foot-over would push the selected swing leg outside the leg workspace, the residual is rejected instead of forcing an unsafe or unreadable trajectory.
- Added RED/GREEN coverage requiring a synthetic low-small case to move root past the obstacle window without activating the repair main path.

## Metrics

Static:

- Focused QP: `58 passed`.
- Pycompile: pass.

Required hard terrain `row=8`, `col=12`, `qp_iterations=2`:

- `viewer_hard_terrain_acceptance_passed=true`.
- `max_playback_readback_error_m≈0.02118`.
- `max_qp_continuous_planned_foot_terrain_penetration_count=0`.
- `max_fk_semantic_collision_count=0`.
- `max_qp_touchdown_on_small_count=0`.
- Remaining FK terrain clearance min `≈-0.00219m`, still inside the existing `5mm` tolerance.

Small obstacle:

- The previous no-op root progress bug is fixed in the synthetic test and the real probe root progress reaches `≈0.36m`.
- The default real crossing probe still fails strict acceptance.
- With the reachability gate, semantic collision and penetration are prevented in the default probe, but `crossing_opportunity_count=0` and playback/FK readback remain high (`≈0.313m`).
- Foot-lane probing showed that forcing lateral foot-over can create FK semantic collision/terrain penetration, so the current solver now rejects unreachable foot-over residuals rather than forcing them.

## Conclusion

This pass fixed the root-cause part where low-small obstacles could fail to produce a crossing window because the root did not progress far enough. It did not complete strict low-small viewer acceptance. The remaining issue is not a candidate-search problem: it is a loss/model tradeoff between foot-over, reachability, IK/FK readback, and semantic safety. The next pass should redesign foot-over as a reachability-aware trajectory loss, not as a stronger hard lateral move.

## Follow-Up

- Keep the low-small root progress residual.
- Redesign the foot-over loss so it first selects reachable swing lanes under fixed gait and only then applies clearance/lift; do not reintroduce endpoint candidates or hard repair.
- Add diagnostics for `qp_continuous_low_small_foot_over_reach_reject_count` to the viewer probe output.
- Re-run small-obstacle default and foot-lane probes after the next loss pass.

## Git Refs

- Baseline Ref: dirty workspace after terrain-clearance and swing-height tuning.
- Candidate Ref: dirty workspace after low-small progress and reachability gate.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
  - `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`
  - `Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py`
