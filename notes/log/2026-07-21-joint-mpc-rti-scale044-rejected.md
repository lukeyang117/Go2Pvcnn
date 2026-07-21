# Joint MPC RTI Scale 0.44 Rejected

## Purpose

Test the single boundary point below scale 0.45 for signed full validity and joint safe margin.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `11.94s`; task GPU memory about `2.09GiB`; no trigger.
- Forward is `24/25` valid with margin `0.1002rad` and passing line-search ratio/run.
- Backward unexpectedly regresses to `22/25` valid despite valid-node margin `0.365rad`.
- Signed tracking, stance, lead, and clearance remain failed.

## Conclusion

Reject 0.44 and stop fine-grained nominal-scale search; active-set branch behavior is non-monotonic. Retain scale 0.45 as the more stable diagnostic base and follow the plan's next order: increase existing contact weight before command tracking pressure.

## Git Refs

- Baseline ref: read-only scale `0.45`, root trust `0.01`
- Candidate ref: read-only scale `0.44`, root trust `0.01`
- Key files: `config.py`, `model/nominal.py`, `solver/trajectory_qp.py`
