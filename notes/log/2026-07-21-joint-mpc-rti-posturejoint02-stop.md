# Joint MPC RTI Posture Joint 0.2 Bracket Stop

## Purpose

Test the single midpoint between posture-joint `0.1` and `0.5`, then stop this parameter sweep if signed validity is not jointly improved.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `12.16s`; task GPU memory about `2.09GiB`; no trigger.
- Forward falls to `11/25` valid; backward is `24/25` valid.
- Backward stance and joint metrics regress heavily; both signed tracking/lead/clearance remain failed.

## Conclusion

Stop posture-joint tuning and restore `0.1`. The scale/trust base fails forward only at late-cycle joint-bound accumulation, so test a small nominal-scale reduction `0.50 -> 0.45` to reserve margin before any command-weight tuning.

## Git Refs

- Baseline ref: read-only scale/trust base, posture joint `0.1`
- Candidate ref: read-only `posture_joint=0.2`
- Key files: `config.py`, `losses/posture.py`
