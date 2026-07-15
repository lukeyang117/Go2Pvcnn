# 2026-07-07 MPC QP Remaining Unit Failures Fixed

## Purpose

Close the non-idle `mpc_qp` unit-test regressions that remained after the idle jitter pass.

## Stage

MPC-QP backend / continuous coupled trajectory solver, decode, diagnostics.

## Related Todo

[T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

- Reproduced the full focused QP suite from the previous `53 passed, 11 failed` state.
- Fixed failures only inside `Go2Pvcnn/extension/batch_mpc_qp_planner/`.
- Kept `mpc` isolated; no edits to `Go2Pvcnn/extension/batch_mpc_planner/` were made for this pass.
- Ran focused regression subsets after each change, then reran the full QP suite and pycompile.

## Key Changes

- Continuous decode now uses unclamped IK for QP output consistency and can emit FK readback for large stance readback errors when root-z readback is enabled.
- Low-small swing samples are lifted in decode before IK when `low_small_swing_clearance_m` requires it, preserving FK/planned consistency.
- Coupled solver now preserves root start XY, allows root-z readback, uses original nominal root progress for non-risk progress preservation, and enforces low-small minimum root progress from the differentiable field target.
- Crossing arc updates now adjust P1/P2 and touchdown endpoint XY/Z directly as optimization variables, not through candidates or lookup.
- Terrain/semantic/body-leg weights and runtime QP parameters were tuned to satisfy semantic, height, body-leg, reachability, and FK readback unit gates.
- Height diagnostics now suppress sub-micrometer numerical deficits.

## Verification

Full QP suite:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Result: `64 passed in 10.07s`.

Pycompile:

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py
```

Result: exit `0`.

Focused checks during the pass included:

- fixed gait stance anchors and touchdown boundary tests: passed
- high low-small arc reachable crossing: passed
- root progress / reachability / FK readback edge tests: passed
- semantic touchdown iteration and low-small swing clearance tests: passed

## Conclusion

The local `mpc_qp` unit-test suite is green. This does not yet claim real IsaacLab viewer acceptance for flat small obstacles or row `8`, col `12`; those still require real `env_isaacsim` probes after this unit pass.

## Git Refs

- Baseline Ref: dirty worktree with full QP suite `53 passed, 11 failed`
- Candidate Ref: dirty worktree after this pass
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/coupled_solver.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/constraints.py`
