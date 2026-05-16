# T302 MPC Body/Leg Collision Design

## Purpose

Record the T302 design pass for adding body/leg/foot height-field collision safety, semantic stance/touchdown obstacle rejection, and high-obstacle velocity/yaw risk scaling to the active MPC backend.

## Stage

Design/spec for `Go2Pvcnn/extension/batch_mpc_planner` after T300e continuous swing-window acceptance.

## Related Todo

- [T302 MPC body/leg height-field collision safety](../todo/T302-mpc-body-leg-height-field-collision-safety.md)
- [T300e continuous swing-window MPC](../todo/T300e-mpc-continuous-swing-window-plan.md)

## Input Conditions

- Baseline ref: `65f0d99` plus existing T300e working tree state documented in [2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md](2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md)
- Candidate ref: working tree with design/todo/log documents
- User requirements:
  - preserve T300e gait/contact/grounding effects;
  - add root/body/knee/shank/swing-foot collision checks from height maps;
  - add semantic touchdown/stance obstacle penalties;
  - cross low semantic small obstacles and avoid high small/large obstacles;
  - include yaw-only obstacle-risk handling;
  - test on `COBBLESTONE_ROAD_CFG` and flat semantic obstacle scenes;
  - use `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python` and real IsaacLab headless tests;
  - keep MPC runtime GPU-only and TDD-driven;
  - add test files only under `Go2Pvcnn/tests/`, and do not add new production files.

## Changes

- Added design spec:
  - [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
- Added T302 branch page:
  - [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)
- Updated dashboard and T300 branch links to include T302.

## Verification

- This is a design-only pass. No implementation tests were run.
- A subagent requirements-coverage review was requested after the spec was written.
- Subagent result: no P0 coverage gaps. P1 clarifications were integrated:
  - small/high-small classification starts from configured semantic small ids, not height alone;
  - risk masks use all configured obstacle cells from scanner tensors, not a one-direction ray;
  - T300e regression acceptance reuses the latest command-matrix/root-cause acceptance baseline.

## Result

The design records T302 as a new branch related to T300e and scopes the future implementation around existing production files plus new tests under `Go2Pvcnn/tests/`.

## Follow-Up

- Integrate subagent review findings.
- Ask the user to review the written spec before producing an implementation plan.
- After user approval, create a TDD implementation plan.

## Git Refs

- Baseline Ref: `65f0d99`
- Candidate Ref: working tree design docs
- Key Files:
  - [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
  - [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)
  - [../todo.md](../todo.md)
  - [../todo/T300-unified-dense-mpc-backend.md](../todo/T300-unified-dense-mpc-backend.md)
  - [index.md](index.md)
