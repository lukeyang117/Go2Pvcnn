# Joint MPC RTI Smooth First 0.20 Diagnostic

## Purpose

Test whether stronger existing first-difference smoothness keeps the frozen phase-matched warm terminal joint compatible with the terminal velocity filter.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Procedure

Ran the ranked three commands for 24 steps with only `loss_terms.smooth_first` changed from `0.02` to `0.20`.

## Result

- Monitor completed in `12.11s`; task GPU memory about `2.09GiB`; no trigger.
- Zero remains fully valid and swing clearance improves from `-0.227mm` to `+0.011mm`; drift worsens to `1.40e-4m`.
- Backward validity improves from `15/25` to `22/25`, supporting a terminal-continuity contribution.
- Forward remains `13/25` valid. Backward still reaches joint margin `0`, alpha-zero run `4`, and has worse stance/yaw/clearance metrics.

## Conclusion

First-difference smoothness helps one symptom but does not close rolling feasibility and causes behavior regressions. Do not adopt yet. Inspect the exact position/velocity filter and worst node/edge at the first forward invalid frame before another parameter point.

## Git Refs

- Baseline ref: working tree on `724a1c3`, smooth first `0.02`
- Candidate ref: read-only config variant `smooth_first=0.20`
- Key files: `config.py`, `losses/smoothness.py`, `runtime/warm_start.py`, `solver/line_search.py`
