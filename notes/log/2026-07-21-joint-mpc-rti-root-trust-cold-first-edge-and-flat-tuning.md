# Joint MPC RTI Root Trust and Cold First Edge

## Purpose

Continue Task 13 flat diagnosis after the zero-command hold amendment and identify the remaining moving-cell blockers.

## Changes

- Root position trust is accumulated by node: `abs(delta_p[:,k]) <= k * root_position_trust`; orientation and joint trust remain constant.
- Cold nominal holds root position and yaw at measured `x0` on edge `x0->x1`; command-scaled integration begins on `x1->x2`.
- Current diagnostic candidate uses `command=20`, `command_linear=2`, `contact=3000`, `step_z=40`, `regularization=0.1`, and `smooth_first=0.05`.

## Verification

- QP/nominal/loss/metrics/RTI focused regression: `61 passed`.
- Monitored CUDA ranked flat, `cuda:1`, 3 cells x 24 steps: completed in about `11.5s`, GPU memory about `2.09GiB`, no watchdog failure.
- Zero command: passed all applicable metrics, zero drift below `1e-6m`.
- Forward `(1,0.5,1)`: velocity `0.140m/s`, direction `0.1743rad`, yaw error `0.159rad`, stance slip `0.222mm`, lead `40ms`, root jump `0.0379m`, validity `1.0`.
- Backward `(-1,-0.5,-1)`: velocity `0.139m/s`, direction `0.167rad`, yaw error `0.161rad`, stance slip `0.223mm`, lead `40ms`, root jump `0.0341m`, validity `1.0`.
- Remaining failures: phase-11 swing surface clearance `-0.0213m/-0.0294m`.

## Conclusion

Accumulated root trust and cold first-edge alignment close the prior tracking, stance, lead, and jump blockers on the ranked representative. Flat is not green because touchdown/swing endpoint clearance remains open. Do not start small or Stage B.

## Git Refs

- Baseline ref: `41f1b18`
- Candidate ref: working tree with uncommitted Task 13 changes
- Key files: `config.py`, `model/nominal.py`, `solver/trajectory_qp.py`, `losses/command.py`, acceptance metrics
