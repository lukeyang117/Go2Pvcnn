# Joint MPC RTI Cold Nominal Scale Geometry Diagnostic

## Purpose

Explain the non-monotonic rolling response to command/step scales by inspecting cold nominal analytic validity, physical joint margin, and adjacent joint step directly.

## Stage

Task 13 nominal root-cause diagnosis; no RTI solve or production edit.

## Procedure

Constructed the ranked B=3 cold nominal once per `step_reference_scale` in `(1.0,0.8,0.6,0.5,0.4,0.3,0.2,0.0)` and reported analytic validity, minimum physical joint margin, maximum adjacent joint step, and position/velocity feasibility.

## Result

- Monitor completed in `5.61s`; task GPU memory about `0.48GiB`; no trigger.
- Zero is analytic-valid and physically feasible for every scale.
- Forward and backward are analytic-invalid for every tested scale.
- Forward has minimum physical margin exactly `-0.837rad` for every scale, indicating an unreachable IK calf clamped to `0rad`; maximum joint step remains `1.42-1.49rad`.
- Backward is position-feasible only at some scales but velocity-infeasible at all scales (`0.99-1.45rad` maximum step).

## Conclusion

Scale tuning cannot repair the cold nominal while analytic reachability remains false. Locate the exact node/leg target geometry and phase event producing unreachable points before any more behavior tuning. This is a nominal formula/geometry audit, not authorization to change the frozen nominal without proving a principled error.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: no code candidate; read-only scale diagnostics
- Key files: `model/nominal.py`, `model/analytic_ik.py`, `model/gait_schedule.py`
