# Joint MPC RTI Kinematic Flat-Small Implementation Plan

## Purpose

Record the replacement implementation plan written after user approval of the 2026-07-20 pure-kinematic flat-small HTML design.

## Plan

- [../../docs/superpowers/plans/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-implementation-plan.md](../../docs/superpowers/plans/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-implementation-plan.md)

The plan has 16 ordered tasks:

1. Freeze H30/24/12+12, seven-loss, compact-state, and five-alpha contracts.
2. Replace adaptive contact with a fixed tensor trot schedule.
3. Add batched analytic IK.
4. Build cold/rolling nominal in one `[B,31,18]` call.
5. Build the unified elevation-semantic field.
6. Replace the objective with exactly seven losses.
7. Linearize directly in `Z`.
8. Add fixed-shape active joint/trust bounds.
9. Add the H30/32 associative scan with dense parity.
10. Replace line search with the approved five-candidate loss-only rule.
11. Route one RTI and delete old production repair paths.
12. Rebuild shared metrics and the monitored heavy-test runner.
13. Close flat first.
14. Close small obstacle second.
15. Freeze and rerun the joint behavior gate.
16. Meet Stage B and freshly rerun behavior plus performance on one final candidate.

## Boundaries

- The old 2026-07-17 implementation plan is superseded as an execution source.
- Semantic preprocessing is fixed grouped Gaussian convolution plus propagated class height, Scharr XY gradients, and bilinear trajectory queries; the optimizer cannot branch on raw semantic ids.
- No new loss, nominal structure, line-search rule, recovery path, output projection, or safety-threshold relaxation is authorized.
- Flat must pass before small; flat+small must pass before Stage B.
- Stage B remains realistic synchronous `1024 x H30 x 1000 <=5.0s`.
- Heavy tests must use the approved process-group heartbeat/timeout/resource watchdog.

## Verification

- Placeholder scan found no `TODO`, `TBD`, or date placeholders.
- The plan contains all fixed design constants and the final behavior/performance rerun.
- Direct-Z GGN parity is specified against residual `J^T J`, not the true nonlinear Hessian.
- Cold and warm nominal share one fixed solver-state input signature.

## Next Step

Choose subagent-driven execution or inline executing-plans. Do not edit production code before selecting the execution workflow and creating an isolated worktree if required by that workflow.
