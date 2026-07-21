# Joint MPC RTI Touchdown And Forward Geometry Diagnostic

## Purpose

Test whether the remaining 48/96-step calf-bound failure comes from a one-node touchdown/stance timing conflict, then isolate the responsible command component without changing the frozen nominal, seven losses, KKT, scan, or line search.

## Phase Result

- Swing is phase `0..11`; `tau=1` is phase `11`; stance starts at phase `12`.
- In both cold and shifted warm nominals, phase-11 and phase-12 touchdown references are identical.
- Nominal FK, phase-12 touchdown reference, and the new stance-onset anchor agree to about `1e-8m`.
- This static sample alone showed no adjacent-node target mismatch; later real rolling evidence below proved the actual conflict was between regenerated touchdown and shifted-warm FK/contact anchor, which made phase-11 Step alignment necessary after the anchor correction.

## Rolling Evidence

With the current time-consistent command and stance-anchor fixes, the ranked `(1,0.5,1)` 48-step trace shows future touchdown targets up to about `0.49m` from root. Their FK error grows to roughly `0.24-0.65m`, while the corresponding calf is repeatedly driven to the physical upper bound `-0.837rad`.

The approved warm contract regenerates gait/map references after shift/rebase but explicitly does not rebuild warm `q` through IK. No warm IK repair was added.

## Parameter And Component Tests

All CUDA runs used the monitored child-process runner.

- `step_reference_scale=0.25/0.5/0.75` is non-monotonic; `0.75` makes signed traces almost entirely invalid, while `0.25/0.5` still reach zero joint margin.
- Reducing Step weight `100 -> 20`, reducing `joint_trust 0.25 -> 0.10`, and increasing `posture_joint 0.1 -> 1 -> 10` do not preserve a positive signed joint margin.
- Pure `vy=0.5` and pure `yaw=1` retain positive margin; pure `vx=1` reaches zero margin/invalidity. Mixed cases inherit the forward failure.
- This narrows the structural audit to command-conditioned forward touchdown/stance geometry, not yaw-frame conversion or phase-12 indexing.

## Metric Correction

`trajectory_valid_ratio` and `map_valid_ratio` now use exact int64 boolean counts instead of float32 means. This prevents CUDA reduction roundoff such as `0.99999994` from rejecting an all-true trace. Metric tests pass `7/7`; the focused nominal/loss/QP/RTI union passed `41/41` before this isolated metric change.

## Proven Reference Fixes

Real rolling context proved the earlier static phase sample incomplete: at future stance onset, regenerated touchdown targets could differ from shifted-warm FK by about `0.29m`. Contact used the shifted FK as anchor while Step used the new target. Two scoped design amendments and RED/GREEN fixes were applied:

1. Current stance at horizon node zero uses current FK; every future stance onset uses its matching touchdown reference and holds it until liftoff.
2. Step applies at the physical swing endpoint phase `11` (`tau=1`), not first stance phase `12`.

Focused regression after both fixes is `43 passed`.

With `step_reference_scale=0.5`, `Step=100`, and the existing diagnostic root/contact configuration, the 48-step ranked result improved to all nodes valid, joint margins `0.141/0.133rad`, stance slip `0.151/0.365mm`, and foot lead `60ms`. Increasing `step_z 0.5 -> 10` makes zero-command swing clearance positive, but signed clearance, root tracking/direction, and joint-step gates remain open. `h_swing=0.12` and terrain weight `20` were rejected because they regress joint step/margin.

## Status

Task 13 remains open. Do not start the formal small-obstacle matrix. Keep the two reference-consistency fixes. Continue only with approved parameter coordination around the current `step_scale=0.5` candidate: signed joint-step smoothing, command tracking/direction, and swing clearance must pass together before a 96-step ranked run.

## Git Refs

- Baseline: `724a1c3` plus current working tree
- Branch: `work/joint-mpc-kinematic`
- Key files: `model/nominal.py`, `losses/step.py`, `losses/contact.py`, `joint_metrics.py`
