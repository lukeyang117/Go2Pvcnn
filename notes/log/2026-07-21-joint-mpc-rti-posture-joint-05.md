# Joint MPC RTI Posture Joint 0.5 Diagnostic

## Purpose

Isolate joint posture regularization from root posture after top-level posture weight 5 failed.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Procedure

Ran the ranked three commands for 24 steps with all top-level weights and trust regions at default, changing only `loss_terms.posture_joint` from `0.1` to `0.5`.

## Result

- Monitor completed in `12.15s`; task GPU memory about `2.09GiB`; no trigger.
- Forward validity falls to `13/25` and stance max slip reaches `3.65mm`.
- Backward remains near the physical joint bound (`5.9e-6rad` margin), stance max slip reaches `20.5mm`, and root tracking/yaw/clearance fail.
- Zero-command drift is `1.26e-4m`, effectively no better than baseline.

## Conclusion

The bound instability is not repaired by stronger joint posture alone. The next approved parameter family is nominal command/reference scale, which can keep the one-RTI linearization point feasible while the unchanged command loss still targets the original command.

## Git Refs

- Baseline ref: working tree on `724a1c3`, posture joint `0.1`
- Candidate ref: read-only config variant `posture_joint=0.5`
- Key files: `config.py`, `losses/posture.py`, `model/nominal.py`
