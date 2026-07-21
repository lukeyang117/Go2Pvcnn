# Joint MPC RTI Scale 0.45 Plus Root Trust 0.01

## Purpose

Reserve late-cycle joint margin relative to scale 0.50 while retaining the informed root trust.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `11.66s`; task GPU memory about `2.09GiB`; no trigger.
- Backward is `25/25` valid, joint margin `0.124rad`, and line-search ratio/run `0.04/1`, all passing.
- Forward improves margin to `0.091rad` and is `24/25` valid; it misses the margin threshold by `0.009rad` and one frame.
- Both signs still fail velocity/yaw or stance, foot lead, swing activity, and clearance. Backward root jump is `0.0725m`.

## Conclusion

Scale 0.45 is near the signed feasibility boundary. Test only `0.44`; if both signs become full-valid with margin >=0.1, freeze this nominal/trust base and move to existing loss tuning.

## Git Refs

- Baseline ref: read-only scale `0.50`, root trust `0.01`
- Candidate ref: read-only scale `0.45`, root trust `0.01`
- Key files: `config.py`, `model/nominal.py`, `solver/trajectory_qp.py`
