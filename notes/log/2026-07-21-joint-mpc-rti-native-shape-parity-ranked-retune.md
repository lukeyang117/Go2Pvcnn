# Joint MPC RTI Native Shape Parity And Ranked Retune

## Purpose

Correct Task 14 formal fixture parity, close the resulting ranked small failures without changing architecture, and recheck the representative real viewer.

## Changes

- Fixed the acceptance field at the real `151x151 @ 0.01m` scanner contract.
- Added grounded native-shape profile coverage at center, half radius, and footprint edge for sphere, cuboid, cylinder, capsule, and cone.
- Corrected formal sphere height from the nominal `0.16m` profile to the grounded `radius=0.06m` sphere surface with `0.12m` peak.
- Corrected formal capsule from a flat disk to its grounded `0.04m` cylinder plus spherical-cap surface.
- Tuned only existing approved parameters: `command_hold_multiplier 2000 -> 4000` and `smooth 14 -> 14.25`.
- Rejected `smooth=16` because it caused `vx=-1` swing clearance `-0.709mm`.

## Evidence

- RED geometry test: sphere and capsule failed; the other three native shapes passed.
- Focused parity/backend: `37 passed`.
- Native-geometry ranked small before tuning: `5/7`; only both pure-yaw cells red.
- A/B: hold `4000` reduced pure-yaw drift to `6.08/7.92um`; smooth `14.25` reduced negative-yaw joint step to `0.34848rad` while preserving negative-forward clearance `+0.726mm`.
- Final ranked small: `7/7` at 160 steps.
- Final ranked flat: `7/7` at 144 steps.
- Full package: `213 passed in 48.70s`.
- `compileall`: pass.
- `git diff --check`: pass.

## Viewer Result

The real S4 sphere `small_forward` event remains red after 49 cycles. Joint step is now green and stance improves, but the remaining failures are:

- root velocity error `0.21852m/s`
- stance stationary ratio `0.9667`
- stance slip max `4.357mm`
- stance anchor residual `4.434mm`
- stance penetration `1.865mm`

Strict crossing, all five collision rates, maximum penetration, touchdown/stance-on-small, airborne touchdown, map/trajectory validity, x0/x1, and cold-once/warm-only remain green. Actual/planned foot readback stays micrometric.

## Result

Fixture parity and ranked behavior are green. Task 14 is not complete because the representative real viewer and the full `29,640`-cell formal matrix remain open. Do not launch formal shards until the viewer stance divergence is diagnosed.

## Viewer Grounding And Worst-Event Follow-Up

- Added viewer JSON diagnostics for the worst continuing-stance XY edge, including phase, leg, foot/root delta, surface error, small distance, alpha, and seven-loss breakdown.
- The old grounding helper aligned foot centers to terrain height while the planner/detector require `terrain + 0.022m`. Added an optional offset with default `0.0` and passed the joint-MPC cfg offset only in this probe.
- The grounding fix removes the false stance penetration failure and improves slip/anchor to `3.173/3.366mm`.
- Current red metrics: root velocity `0.22152m/s`, stationary ratio `0.9778`, swing clearance `-1.958mm`, stance slip/anchor `3.173/3.366mm`.
- Worst edge: FL phase `20->21`; root `+21.60mm`, foot longitudinal `-3.08mm`, small distance `59->62mm`. It is continuing stance and moving away from the obstacle.
- Contact-anchor A/B `400->800` reduces slip to `1.415mm` but worsens root velocity/stationary; it is rejected and config is restored to `400`.
- Post-follow-up regression: joint package plus viewer reset suite `248 passed`; compileall passes.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan: [Joint MPC RTI pure-kinematic implementation plan](../../docs/superpowers/plans/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-implementation-plan.md)
