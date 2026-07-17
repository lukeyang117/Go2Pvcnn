# Joint MPC Root-Joint Coupled Gait Implementation Plan

- Purpose: record the executable TDD plan approved after the coupled-gait HTML design.
- Stage: T302v.7 plan; implementation authorized inline until Stage C passes.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `2a2dbe7`.
- Candidate Ref: plan working tree.
- Plan: `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md`.

## Structure

- Stage A tasks 1-8: shared JointMetrics, complete FK Jacobian, correct coupled LQ, stance equality, horizon command/touchdown, foot-leading-root, scenario-metric Cartesian gate, and full behavior closure.
- Stage B tasks 9-12: frozen workload profile, exact batched EDT or coupled Schur optimization according to measured bottleneck, and formal `<=5s` closure.
- Stage C task 13: fresh behavior plus performance rerun on one final candidate.

## Result

- Plan self-review passes with 13 concrete tasks and no placeholders.
- User already authorized inline execution after the plan.
