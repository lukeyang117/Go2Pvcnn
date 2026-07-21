# Joint MPC RTI Flat Soft-Objective Contract Blocker

## Purpose

Continue Task 13 after the rolling-time command amendment and determine whether the ranked flat gate can be closed by the approved scalar tuning surface.

## Stage

Pure-kinematic H30 ranked flat behavior diagnosis on `work/joint-mpc-kinematic`. This is not flat acceptance evidence.

## Procedure

- Focused loss/nominal/QP/line-search/RTI/flat-metric regression: `55 passed`.
- Every behavior point ran through `run_monitored_joint_mpc.py` on `cuda:1`, with three ranked commands, 24 steps, a 120-second timeout, and a 5-second heartbeat.
- Runs completed in about 11-12 seconds with about 2.09 GiB task GPU memory and no watchdog trigger.
- H30, gait `24/12+12`, seven losses, two active refinements, five alphas, three filters, and loss-only selection remained fixed.

## Evidence

The feasible base was made explicit in production config: regularization `0.1`, command scale `0.45`, root trust `0.01`, contact anchor/ground `200/32`, and first-edge-only command early weight `0`.

With touchdown scale `1` and command weight `1`, all ranked rows were valid, but zero drift was `3.58 mm`; signed root velocity error was `0.296/0.331 m/s`, and signed swing endpoint clearance was `-79.4/-56.8 mm`.

Changing only touchdown scale to `0.5` preserved validity and improved signed swing clearance to `-2.64/-16.1 mm`, but root velocity error became `0.453/0.506 m/s`. Changing only command weight from `1` to `5` improved zero drift to `0.829 mm` and signed yaw error to `0.241/0.254 rad/s`; signed root velocity and clearance still failed.

Increasing root trust from `0.01` to `0.02` did not recover tracking and regressed stance slip, stance stationary ratio, anchor residual, and foot-lead upper bound. It was rejected. Restoring root command scale to `1.0` while keeping touchdown scale `0.5` recovered tracking on valid samples, but forward validity collapsed to `0.04` and backward foot lead/leak and joint margin failed. It was rejected.

Increasing command weight to `100` reduced zero drift only to `46.2 um`, still above the frozen `10 um` threshold, while zero-command stance root-carry ratio regressed to `0.909`; moving tracking and clearance still failed. It was rejected.

At the zero-command cold nominal, command loss is exactly zero while contact, posture, terrain, swing-speed, and smooth losses are nonzero. The initial default FK feet are `6.69 mm` above the configured contact surface. Loss-only selection therefore has a legitimate incentive to accept a small root/joint adjustment even for zero command. Increasing the soft command weight trades drift against stance behavior but does not satisfy both frozen gates.

## Conclusion

The ranked flat failure is no longer explained by map gradients, scan parity, active-set feasibility, line-search filtering, warm terminal copying, or a missing scalar point. The remaining contract asks one purely soft, scenario-independent seven-loss objective to guarantee both `root_zero_drift_m <= 1e-5` and moving root/stance/swing behavior. Tested approved scalar directions move failures between those gates rather than closing them.

Stop scalar sweeping. Before further production edits, make an explicit design decision about the zero-command contract: either define a continuous command-activity weighting that can be proved compatible with moving command pressure, or revise the acceptance threshold to a physically justified soft-objective tolerance. Do not add a zero-command branch, output projection, recovery path, eighth loss, or hidden candidate filter.

## Current Candidate

- `regularization=0.1`
- `command_scale=0.45`
- `step_reference_scale=0.5`
- `root_position_trust=0.01`
- `contact_anchor_xy=200`
- `contact_ground=32`
- `command_early_swing=0`
- `command=5`

This is the most balanced diagnostic base, not an accepted Task 13 configuration.

## Git Refs

- Baseline Ref: `724a1c3`
- Candidate Ref: working tree on `724a1c3`
- Key Files: `config.py`, `losses/command.py`, `model/nominal.py`, `runtime/warm_start.py`, `solver/trajectory_qp.py`, `joint_metrics.py`
