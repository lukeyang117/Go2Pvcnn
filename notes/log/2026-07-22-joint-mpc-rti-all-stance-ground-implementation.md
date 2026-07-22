# Joint MPC RTI All-Stance Ground Implementation

## Purpose

Implement the user-approved Task 14E-Z方案 A: pre-LQ warm nominal grounding for every published x1 stance leg, plus all-x1-stance raw-ground feasibility inside the existing fourth line-search filter.

## Implemented Contract

- Continuing stance target: persistent-anchor XY plus raw terrain height and `foot_contact_offset`.
- Touchdown onset target: shifted-x1 FK XY plus raw terrain height and `foot_contact_offset`.
- Only x1 stance joints are replaced through existing batched analytic IK.
- x0, x1 root/RPY, x1 swing joints and x2..x30 remain unchanged.
- Existing warm-only invalid semantics, physical joint bounds and `30rad/s * 0.02s` edge bound remain active.
- `published_kinematics` now combines continuing XY, all-stance raw-ground and swing safe-floor checks without adding a filter, scalar, QP or RTI iteration.

## TDD And Static Verification

- Nominal RED reproduced continuing/onset z above raw ground.
- Line-search RED failed on the missing stance-ground mask API.
- Nominal and line-search GREEN: `29 passed` and `15 passed`.
- Focused nominal/loss/QP/scan/line-search/RTI/backend: `127 passed`.
- Contract/analytic-IK/gait/kinematics/terrain: `47 passed`.
- `py_compile` and `git diff --check`: pass before the representative viewer.

## Representative Viewer

- Scenario: real S4 sphere `small_forward`.
- Device: `cuda:1`.
- Report: `/tmp/joint_mpc_viewer_all_stance_ground_diag.json`.
- Executed cycles: `48`; actual samples: `49`.

Green metrics include root velocity `0.10981m/s`, root direction `0.08412rad`, stance XY slip `0.01893mm`, positive swing clearance, zero part collisions, zero touchdown/stance on small, strict crossing `1.0`, and cold/warm lifecycle `1/47`.

The representative gate remains RED:

- trajectory validity `0.97959`;
- stance ground gap `100.75mm`;
- airborne touchdown `0.020408`;
- joint step `0.59164rad`;
- one phase-23 cycle has all five `published_kinematics` candidates false.

## Root Cause

At the worst post-obstacle touchdown onset, the approved shifted XY queries a raw small-obstacle surface approximately `100.26mm` above ordinary ground. The manifold therefore changes the foot center only from `127.95mm` to `122.26mm`; it correctly grounds to the obstacle top rather than finding a safe ground XY. On the next continuing-stance cycle, the target surface is ordinary ground and the nominal drops approximately `100.75mm`, producing the `0.59164rad` joint step.

The all-alpha phase-23 failure is the same discrete-switch conflict: directions that satisfy the new swing floor perturb new stance ground, while alpha zero keeps the new swing below its floor. No approved alpha satisfies both parts of the fourth filter.

## Decision

Retain the approved implementation as explicit evidence, but do not claim behavior closure. Task 14F, ranked/formal and Stage B remain blocked. The next behavior design must choose safe touchdown XY ownership or pre-touchdown descent timing; do not relax ground, joint, validity or filter thresholds and do not tune scalar weights inside Task 14E-Z.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `work/joint-mpc-kinematic` plus uncommitted Task 14E/14E-Z work
- Design: `docs/superpowers/specs/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-design.html`
- Plan: `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md`
