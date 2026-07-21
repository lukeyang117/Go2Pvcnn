# Joint MPC RTI Scale/Trust With Posture Joint 0.5

## Purpose

Test whether stronger joint posture keeps the forward calf away from its late-cycle bound on the feasible `command_scale=0.5`, `root_position_trust=0.01` base.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `11.48s`; task GPU memory about `2.09GiB`; no trigger.
- Forward becomes `25/25` valid and line alpha-zero ratio passes at `0.04`, but minimum joint margin remains nearly zero (`3.1e-5rad`), stance and root jump fail, and velocity error is `0.346m/s`.
- Backward regresses to `22/25` valid despite a healthy valid-node margin `0.211rad`; tracking, stance, lead, and clearance fail.
- Zero drift improves slightly to `1.08e-4m` but still fails.

## Conclusion

Joint posture changes the rolling branch but is directionally non-monotonic. Test one bracket midpoint `posture_joint=0.2`; if it cannot keep both signed commands fully valid, stop sweeping this parameter.

## Git Refs

- Baseline ref: read-only scale/trust base (`posture_joint=0.1`)
- Candidate ref: read-only `posture_joint=0.5`
- Key files: `config.py`, `losses/posture.py`, `solver/trajectory_qp.py`
