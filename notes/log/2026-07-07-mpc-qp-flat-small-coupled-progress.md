# 2026-07-07 MPC-QP Flat-Small Coupled Progress

## Purpose
Continue T302v flat-ground small-obstacle optimization with the hard constraints that `mpc_qp` remains fully coupled, no candidate endpoint search, no touchdown lookup, and no post-QP repair in the continuous main path.

## Stage
MPC-QP backend / continuous Bezier trajectory / flat-small real viewer probe.

## Related Todo
[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Code Changes
- `Go2Pvcnn/extension/batch_mpc_qp_planner/coupled_solver.py`
  - Anchored low-small target detection to the nominal/root-start XY so active masks do not disappear after intermediate iterations.
  - Added existing-design crossing residual refinements: midpoint along loss, persistent sample/FK crossing height/lane loss, and continuity/root-z soft losses.
  - Kept updates inside the single shared `total` loss over foot controls, root, and rpy.
  - Kept `P0` fixed during the QP update; it is a boundary condition, not an optimization variable.
- `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
  - Fixed `build_controls_from_nominal()` so `P0` preserves the current/nominal foot 3D position instead of binding start-foot z to terrain. `P3/touchdown z` remains terrain-bound.

## Verification
Focused pytest:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_low_small_qp_creates_crossing_leg_from_trajectory_loss Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_continuous_low_small_high_arc_remains_fk_reachable -q
```

Result: `1 passed, 1 failed`. The 0.05m flat-small crossing regression still passes; the 0.16m synthetic high-small regression still fails at `qp_iterations=2`.

Real GPU2 flat-small probes used:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations N
```

Key metrics:

- Before P0-z fix, best observed `qp_iterations=3`: foot-over success `1`, semantic collision `0`, terrain penetration `0`, readback `0`, but foot jump `≈0.3500m` and joint jump `≈1.311rad`, so acceptance failed.
- Diagnosis showed the worst foot jump was frame `0 -> 1`, leg `2`: planned frame-0 foot had been terrain-bound below the real current foot. This violated the continuous boundary condition.
- After preserving `P0.z`, `qp_iterations=4` reached foot-over success `1`, semantic collision `0`, terrain penetration `0`, but still failed: foot jump `≈0.28145m`, joint jump `≈3.135rad`, FK low-small swing height `≈0.1867m`.
- After preserving `P0.z`, `qp_iterations=5` reached foot-over success `1` and foot jump `≈0.2446m`, but had semantic collision `5`, joint jump `≈3.125rad`, and FK low-small swing height `≈0.1849m`.
- Joint-jump diagnosis showed leg `3` thigh flips from near `0` to `≈3.13rad` while calf is clamped at the lower limit, indicating IK branch/limit discontinuity rather than planned-vs-FK readback mismatch.

## Result
Partial. The main path remains coupled and repair-free, and the P0 boundary bug is identified/fixed. Flat-small is not yet accepted because solving the foot-over objective still creates IK joint discontinuity or semantic collision depending on `qp_iterations`.

## Follow-Up
Next work should target IK branch/limit continuity inside the coupled objective or a differentiable leg-selection/phase weighting so the crossing leg remains reachable and continuous. Do not add candidates, touchdown lookup, hard constraints, or repair fallback.

## Git Refs
- Baseline Ref: current dirty workspace before this pass.
- Candidate Ref: current dirty workspace after `coupled_solver.py` and `continuous.py` edits.
- Key Files: `Go2Pvcnn/extension/batch_mpc_qp_planner/coupled_solver.py`, `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`, `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`.
