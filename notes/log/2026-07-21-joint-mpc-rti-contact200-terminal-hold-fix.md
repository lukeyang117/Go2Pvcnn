# Joint MPC RTI Contact 200 Diagnostic And Warm Terminal Hold Fix

## Purpose

Continue Task 13 in the approved tuning order: test stronger contact pressure, then diagnose the resulting candidate infeasibility instead of sweeping more weights.

## Stage

Ranked flat 3-cell x 24-step diagnosis and a principled warm-nominal correction.

## Evidence

- Read-only `contact=200` on the scale `0.45`, root-trust `0.01` base regressed forward/backward validity to `22/25`; forward joint margin fell to `0.020rad`. Reject this weight.
- At the original `contact=100` base, the sole invalid frame was forward frame 12. Every alpha failed joint velocity because the alpha-zero nominal itself had a `0.689rad` node-29-to-30 jump.
- Exact source: leg-4 thigh changed `0.572212 -> 1.261264rad`. Warm shift appended old `q7`, while new node 29 was optimized old `q30`; the optimizer has no periodic `q30=q6` equality, so the copy invented a discontinuity after an otherwise filter-feasible accepted trajectory.

## TDD And Change

- RED proves an accepted trajectory with maximum joint edge `0.35rad` became `0.70rad` only after shift.
- Warm terminal root still extrapolates; warm terminal joints now hold the previous accepted terminal joint.
- The stale exact `q30=q6` test was replaced by the accepted-terminal hold contract.
- Nominal suite: `12 passed`.
- Focused flat/metrics/line-search/RTI/nominal/terrain regression: `63 passed`.
- One first regression command referenced nonexistent `test_terrain.py` and collected no tests; the corrected command used `test_terrain_fields.py`.

## Post-Fix Ranked Result

- Monitor completed in `11.51s`, task GPU memory about `2.09GiB`.
- Zero, forward, and backward are all `25/25` valid; all line-search fallback metrics pass.
- Forward joint margin is only `5.0e-5rad`; backward margin is `0.248rad`.
- Flat behavior remains open: stance, foot lead, tracking, clearance, zero drift, and root jump still contain failures.

## Conclusion

The terminal hold removes the rolling velocity-filter defect without changing H30, gait, KKT, losses, candidates, or filters. Re-test the approved smaller joint trust now that the previous terminal confound is gone; do not increase contact again.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `runtime/warm_start.py`, `test_nominal.py`, `config.py`
