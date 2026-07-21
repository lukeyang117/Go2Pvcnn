# Joint MPC RTI Step Reference Scale 0.50 Diagnostic

## Purpose

Test whether reducing only the approved command-conditioned touchdown lead makes the cold nominal feasible for one RTI.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `11.78s`; task GPU memory about `2.09GiB`; no trigger.
- Forward validity collapses to `3/25`; joint margin/step, penetration, yaw, lead, and clearance fail.
- Backward becomes `25/25` valid but reaches joint margin `0`, joint step `0.390rad`, alpha-zero run `6`, stance slip `47.8mm`, and large stance Z/yaw failures.
- Zero is unchanged.

## Conclusion

Reject `step_reference_scale=0.5`. The response is non-monotonic because cold nominal switches between its full and reduced analytic-IK branches. Before another rolling parameter run, inspect cold nominal physical bounds/velocity and analytic validity directly across scales.

## Git Refs

- Baseline ref: working tree on `724a1c3`, step scale `1.0`
- Candidate ref: read-only config variant `step_reference_scale=0.5`
- Key files: `model/nominal.py`, `model/analytic_ik.py`, `config.py`
