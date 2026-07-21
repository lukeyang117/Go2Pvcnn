# Joint MPC RTI Joint Trust 0.20 Diagnostic

## Purpose

Check whether a smaller reduction than `0.15` preserves RTI correction ability while keeping the backward trajectory away from joint bounds.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Procedure

Ran the ranked three commands for 24 steps on `cuda:1` with only `joint_trust=0.20` changed.

## Result

- Monitor completed in `12.06s`; task GPU memory about `2.09GiB`; no trigger.
- Forward remains only `14/25` valid, with yaw, stance, lead, and clearance failures.
- Backward margin remains improved at `0.168rad` and stance slip passes at `0.254mm`, but alpha-zero ratio/run worsens to `0.591/10`; valid count is `22/25`.
- Zero-command drift worsens to `1.94e-4m`; clearance is `-0.264mm`.

## Conclusion

`0.20` does not provide a useful compromise and must not replace the default. Restore `joint_trust=0.25` and test the next approved tuning direction, existing posture weight, as one variable.

## Git Refs

- Baseline ref: working tree on `724a1c3`, default trust `0.25`
- Candidate ref: read-only config variant `joint_trust=0.20`
- Key files: `config.py`, `solver/trajectory_qp.py`, `run_joint_acceptance.py`
