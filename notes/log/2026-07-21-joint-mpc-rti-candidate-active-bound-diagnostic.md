# Joint MPC RTI Candidate And Active-Bound Diagnostic

## Purpose

Determine whether the first B1/B3 backward alpha divergence is a line-search tie, a filter difference, or an active-QP direction discontinuity.

## Stage

Task 13 root-cause diagnosis before approved parameter tuning.

## Procedure

Wrapped the existing five-candidate line search in a temporary read-only diagnostic and ran backward B=1 then ranked B=3 for 12 rolling steps under the monitored CUDA runner. Frames 9-12 printed all candidate losses, finite/position/velocity filters, selected index, nominal minimum joint margin, and maximum scan direction magnitude.

## Result

- Monitor completed in `10.49s`; task GPU memory reached about `2.25GiB`; no watchdog trigger.
- This is not a line-search tie. At B1 frame 10, alpha `1.0` has the lowest loss `4.64` but fails position and velocity, so alpha `0.5` with loss `7.29` is selected.
- At B3 frame 10, alpha `1.0` and `0.5` fail position, so alpha `0.25` is selected; the scan direction max is `2.95`, versus `0.78` in B1.
- Both frame-10 nominals already contain at least one future joint exactly on a physical bound (`nominal_margin=0`). Small preceding state differences therefore change the active set and produce a large next direction difference.
- The line search uses only the approved five alphas, three filters, and seven-loss score. No extra filter or ranking behavior was observed.

## Conclusion

The rolling sensitivity originates before line-search scoring, at a nominal trajectory sitting on active joint bounds. One exact-repeat scan parity check remains before following Task 13's approved first tuning direction: reduce trust region, then increase posture weight if needed to restore joint safe margin.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: working tree on `724a1c3`
- Key files: `solver/trajectory_qp.py`, `solver/trajectory_scan.py`, `solver/line_search.py`, `solver/sqp_rti.py`
