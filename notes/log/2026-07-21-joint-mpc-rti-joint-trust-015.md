# Joint MPC RTI Joint Trust 0.15 Diagnostic

## Purpose

Test the approved first Task 13 tuning direction after proving exact-repeat scan parity: reduce only `joint_trust` from `0.25` to `0.15`.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config was not edited.

## Procedure

Ran the three ranked commands for 24 steps on `cuda:1` under the monitored runner. Every parameter except `cfg.solver.joint_trust=0.15` remained at the current default.

## Result

- Monitor completed in `11.64s`; task GPU memory about `2.09GiB`; no trigger.
- Backward joint safe margin improves from `0` to `0.168rad`; stance max slip improves from `50.9mm` to `0.071mm`; stance gap and yaw tracking pass.
- Backward still fails joint step at `0.416rad`, alpha-zero ratio/run at `0.458/10`, root lead, and swing clearance `-37.9mm`.
- Forward validity drops to `14/25`; alpha-zero ratio improves to `0.071` but yaw error, stance anchor/slip, root lead, and clearance fail.
- Zero-command drift worsens from `1.24e-4m` to `1.57e-4m`; clearance remains about `-0.23mm`.

## Conclusion

Reducing joint trust moves the backward trajectory away from the physical bound and repairs stance behavior, confirming the diagnosis, but `0.15` is too restrictive for the forward and fallback gates. Test the smaller reduction `0.20` before changing posture weight.

## Git Refs

- Baseline ref: working tree on `724a1c3`, default trust `0.25`
- Candidate ref: read-only config variant `joint_trust=0.15`
- Key files: `config.py`, `solver/trajectory_qp.py`, `run_joint_acceptance.py`
