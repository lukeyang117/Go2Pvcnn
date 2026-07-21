# Joint MPC RTI Liftoff Reference Phase Fixes

## Purpose

Correct two provable phase/formula errors in cold nominal swing lift endpoints that produced command-sized foot-reference jumps and infeasible joint velocity edges before LQ.

## Stage

Task 13 principled nominal correction within the approved exception boundary.

## Root Causes

1. A leg at exact phase-zero liftoff used command-led event placement because `lift_raw == 0` missed the inferred/measured branch. Its x0 foot reference differed from measured FK by up to `0.12m`.
2. A future stance-to-swing transition built lift from `root_at_lift + footprint + lead` instead of the preceding stance/touchdown foot. The largest foot-reference discontinuity was `0.356m`; at command scale `0.4`, leg-2 thigh jumped `0.736rad` across node 11->12.

## TDD Evidence

- Phase-zero RED: `1 failed, 9 passed`; GREEN after changing the exact boundary to measured/inferred lift: `10 passed`.
- Future-liftoff RED: `1 failed, 10 passed`; GREEN after vectorized previous-touchdown association: `11 passed`.
- No environment/time/leg loop was introduced.

## Change

- Exact current liftoff (`lift_raw <= 0`) uses the inferred lift endpoint that reconstructs measured x0.
- Future lift uses the previous touchdown event one stance half-cycle earlier; if that event is at/before horizon start, it inherits measured foot.
- Touchdown construction, stance construction, H30/24/12+12 schedule, seven losses, KKT, five candidates, and line-search filters are unchanged.

## Post-Fix Geometry

The repeated cold command-scale diagnostic shows `command_scale=0.5` makes all ranked cold nominals analytic/position/velocity feasible, with forward/backward max joint steps `0.330/0.334rad`; before the future-lift fix, corresponding steps remained above the `0.6rad` filter for forward.

## Result

Scoped nominal pass. A broader regression and rolling ranked run remain required; flat is still open.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `model/nominal.py`, `test_nominal.py`
