# Joint MPC RTI Support Candidate Diagnostics

## Purpose

Identify why the grounded affine support candidate still fails one validity node, root tracking, and swing clearance before authorizing another behavior change.

## Diagnostic Change

- Added fixed-shape `candidate_filter_valid[B,5,4]` and `candidate_loss[B,5]` outputs to the existing five-alpha line search.
- Propagated those tensors and `support_target[B,6]` through the one-RTI and step diagnostics only.
- Added invalid-node and metric-applicable worst-swing solver layers to the real viewer report.
- No candidate, filter, objective, KKT row, alpha, selection rule, or published state changed.

## Verification

- RED: missing line-search filter tensor and step diagnostics fields.
- GREEN: scoped `2 passed`.
- Focused QP/scan/line-search/RTI/backend regression: `63 passed`.
- Real report: `/tmp/joint_mpc_viewer_support_diagnostic_v3.json`.

## Evidence

The only invalid node is global phase 12. All five candidates pass all four filters. Candidate losses for alpha `(1,0.5,0.25,0.125,0)` are:

```text
(1000.94, 251.37, 64.78, 18.52, 3.62)
```

The affine onset target asks the two newly-stance feet to move approximately `26-32mm` in XY in one x1 step. The seven-loss merit therefore selects alpha zero. The selected nominal is analytically invalid, so publication receives status 1. This is not a line-search filter failure.

The formal worst swing-clearance event is node 36. Foot-center z is:

```text
nominal x1 = 17.446mm
full QP x1 = 22.602mm
selected x1 = 18.704mm
contact surface = 22.000mm
```

All candidates again pass all filters. Candidate losses are `(31.24,4.28,2.68,3.10,3.79)`, so alpha `0.25` is selected and leaves `-3.296mm` clearance. The grounded full-XYZ support correction changes the full-horizon direction enough that the loss-only line search trades away physical swing clearance and root tracking.

## Architecture Conclusion

The next minimal candidate should redefine the hard support contract, not add scalar tuning:

- hard KKT rows cover world XY only;
- continuing stance target is persistent-anchor XY correction;
- x1 touchdown onset uses zero target and preserves nominal XY instead of forcing a stale/large affine landing correction;
- Z remains in the existing Contact-ground and Terrain objective;
- the existing exact-FK continuing-stance XY filter remains unchanged.

This keeps full 18D root-joint optimization, one QP, one RTI, five alphas, seven losses, and fixed rank. It requires explicit user approval because it changes the approved six-row support architecture.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `solver/line_search.py`, `solver/sqp_rti.py`, `types.py`, `planner.py`, `joint_mpc_rti_viewer_reproduction_probe.py`
