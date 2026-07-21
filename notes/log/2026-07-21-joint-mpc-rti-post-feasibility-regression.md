# Joint MPC RTI Post-Feasibility Regression And Ranked Baseline

## Purpose

Verify the infeasible-fallback status fix broadly and establish the default ranked failure surface before further approved tuning.

## Stage

Task 13 flat behavior diagnosis.

## Verification

- Focused union across flat aggregation, metrics, line search, RTI pipeline, nominal, and terrain: `59 passed in 9.28s`.
- Fresh default ranked 3-cell x 24-step CUDA run completed under the watchdog in `11.64s`, task GPU memory about `2.09GiB`.

## Ranked Result

- Zero: `25/25` valid; drift `1.24e-4m` and swing clearance `-0.227mm` fail.
- Forward: `13/25` valid; root lead/leak, yaw, swing activity, and clearance fail.
- Backward: `15/25` valid; root lead/leak, yaw, stance anchor/slip/stationary ratio, swing activity, and clearance fail.
- Infeasible alpha-zero frames are no longer reported as valid. This directly motivated the universal `trajectory_valid_ratio=1.0` metric added in the subsequent TDD pass.

## Conclusion

The fix is regression-green and exposes a real rolling feasibility failure rather than masking joint-bound frames. The next hypothesis is warm terminal joint continuity: the frozen phase-matched terminal copy can violate the terminal velocity edge unless existing smoothness keeps node 29 and node 30 compatible.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `solver/line_search.py`, `solver/sqp_rti.py`, `joint_metrics.py`, `run_joint_acceptance.py`
