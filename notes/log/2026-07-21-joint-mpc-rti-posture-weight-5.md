# Joint MPC RTI Posture Weight 5 Diagnostic

## Purpose

Test the second approved Task 13 joint-bound tuning direction after rejecting reduced joint trust: increase only the existing top-level posture weight from `1` to `5`.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Procedure

Ran the ranked three commands for 24 steps on `cuda:1` with default `joint_trust=0.25` and only `losses.posture=5.0` changed.

## Result

- Monitor completed in `11.49s`; task GPU memory about `2.09GiB`; no trigger.
- Backward joint margin improves to `0.341rad` and joint step passes at `0.166rad`.
- Forward and backward valid counts fall to `14/25` and `12/25`; both retain lead, yaw, stance/clearance, and alpha-zero failures.
- Zero-command drift worsens to `2.93e-4m`; zero swing clearance improves to `-0.036mm` but still fails strict nonnegative clearance.

## Conclusion

Increasing the whole posture family overweights root height/roll-pitch together with joints and does not preserve rolling validity. Do not adopt. Isolate the already-approved `posture_joint` subweight next while keeping all seven top-level weights and root posture subweights unchanged.

## Git Refs

- Baseline ref: working tree on `724a1c3`, posture `1.0`
- Candidate ref: read-only config variant `posture=5.0`
- Key files: `config.py`, `losses/posture.py`, `run_joint_acceptance.py`
