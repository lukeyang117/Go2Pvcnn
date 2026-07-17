# Joint MPC Root-Joint Coupled Gait Implementation Plan

- Purpose: record the executable TDD plan approved after the coupled-gait HTML design.
- Stage: T302v.7 plan; implementation authorized inline until Stage C passes.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `2a2dbe7`.
- Candidate Ref: plan working tree.
- Plan: `docs/superpowers/plans/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-implementation-plan.md`.

## Structure

- Stage A tasks 1-9: shared JointMetrics, complete FK Jacobian, correct coupled LQ, stance equality, horizon command/touchdown, foot-leading-root, scenario-metric Cartesian gate, H16-H50 full-cycle/RTI-direction exploration, and full behavior closure on the shortest passing `H_selected`.
- Stage B tasks 10-13: frozen selected-horizon profile, exact batched EDT when field dominates, MPX-referenced associative TVLQR/root-Schur/parallel-line-search optimization, and formal `1024 x H_selected x 1000 <=5s` closure.
- Stage C task 14: fresh behavior plus performance rerun on one final candidate and the same `H_selected`.

## Result

- Plan amendment self-review passes with 14 concrete tasks and no placeholders.
- User already authorized inline execution after the plan.
