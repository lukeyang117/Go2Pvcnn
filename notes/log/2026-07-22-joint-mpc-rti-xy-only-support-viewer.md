# Joint MPC RTI XY-Only Support Viewer

## Purpose

Implement and evaluate the user-approved four-row world-XY support amendment after candidate diagnostics rejected the six-row XYZ contract.

## Stage

- Planner: `Go2Pvcnn/extension/joint_mpc_rti`
- Gate: Task 14 representative S4 sphere `small_forward`
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Contract Change

- `support_jacobian` changes from `[B,6,18]` XYZ rows to `[B,4,18]` world-XY rows for the two x1 stance feet.
- Continuing stance uses `persistent_anchor_xy - nominal_foot_xy`.
- Touchdown onset uses a zero affine target, preserving nominal x1 XY instead of forcing the current touchdown correction through KKT.
- Z remains in the existing Contact-ground and Terrain residuals.
- Full 18D variables, one QP, one RTI, five alphas, four filters, seven loss families, and cold-once/warm-only remain unchanged.

## TDD And Focused Verification

- RED: four support tests failed on the old six-row shape and onset target.
- GREEN: four-row Jacobian/FK parity, continuing affine target, zero onset target, and dense KKT feasibility passed.
- RED/GREEN: planner diagnostic fallback changed from six to four entries.
- Split focused regression: `14 + 12 + 9 + 31 = 66 passed` across QP, scan, RTI pipeline, line search, backend wiring, and contract tests.
- Original box, joint-position, and joint-velocity bounds remain covered and green.
- `git diff --check`: pass before the viewer run.

## Real Viewer Evidence

The actual-state S4 sphere `small_forward` case ran on `cuda:1` with direct published-x1 playback, the real scanner field, the shared detector, and 49 executed RTI cycles. Report: `/tmp/joint_mpc_viewer_support_xy_only.json`.

| Metric | Six-row grounded XYZ | Four-row XY-only | Threshold | Result |
| --- | ---: | ---: | ---: | --- |
| trajectory valid ratio | `0.98` | `1.0` | `1.0` | improved, pass |
| joint step | `0.34094rad` | `0.31995rad` | `<=0.35rad` | pass |
| stance slip / anchor | `0.136/0.181mm` | `0.164/0.164mm` | `<=0.5mm` | pass |
| root velocity error | `0.21931m/s` | `0.22483m/s` | `<=0.2m/s` | regress, fail |
| swing clearance | `-3.296mm` | `-2.593mm` | `>=0` | improved, fail |
| cold / warm lifecycle | one invalid cycle | `1 cold + 48 warm`, no restart | exact | pass |

All collision, semantic, map, strict-crossing, stance-stationary, joint-limit, nonfinite, and penetration gates pass. The worst swing node selects alpha `1.0`; its nominal/full/selected foot-center z is `18.620/19.407/19.407mm` against a `22mm` surface. Candidate losses increase monotonically from alpha one to zero, so the remaining clearance failure is again selected by the seven-loss merit rather than caused by a line-search filter.

## Conclusion

The four-row contract removes the touchdown-onset validity failure and improves joint continuity and clearance, but it does not close representative acceptance. Root tracking regresses and swing clearance remains negative. Do not run ranked/formal matrices or tune another scalar/support-anchor variant. The next behavior change requires a new architecture decision about explicit swing feasibility or merit handling under one RTI.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `solver/linearization.py`, `planner.py`, `test_trajectory_qp.py`, `test_trajectory_scan.py`, `test_rti_pipeline.py`
