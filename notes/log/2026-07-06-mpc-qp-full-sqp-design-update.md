# MPC QP Full SQP/QP Design Update

## Purpose

Record the corrected `mpc_qp` design direction after user clarification: touchdowns must be optimized by a full fixed-shape SQP/QP formulation, not found by candidate search, nearby lookup, ring repair, or endpoint line-search.

## Stage

MPC-QP backend / full SQP-QP trajectory design and todo alignment.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Documentation-only update:

- Updated [../../docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html](../../docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html).
- Updated [../todo.md](../todo.md).
- Updated [../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md).

## Key Decisions

- `touchdown_xy` is a QP decision variable and must be solved, not searched.
- `touchdown_z` is bound through a differentiable height field: `z = h(xy)` and local QP linearization uses `h0 + grad_h^T delta_xy`.
- Discrete height and semantic maps should be converted to differentiable fields before the QP loop:
  - height `h, grad_h`
  - semantic risk `s, grad_s`
  - roughness/edge/support `r, grad_r`
  - clearance field or equivalent fixed sampled residuals
- Alternating diagonal gait must be encoded by fixed stance/swing masks and QP constraints.
- Candidate endpoint line-search, fixed-offset touchdown selection, ring search, nearby safe-point lookup, and hard repair are forbidden as the future `mpc_qp` main path.
- Required viewer acceptance terrain: `--terrain-row 8 --terrain-col 12` with the user's viewer command; do not hard-code `CUDA_VISIBLE_DEVICES`.

## Result

Documentation aligned. No code implementation or verification was claimed in this log.

## Follow-Up

Create/update the implementation plan so Task 18 removes or bypasses candidate/search/repair scaffolding from the `mpc_qp` main path and introduces fixed-shape SQP/QP modules for fields, gait masks, variables, QP assembly, and solver.

## Git Refs

- Baseline Ref: local working tree before design correction
- Candidate Ref: local working tree after documentation/todo update
- Key Files:
  - `docs/superpowers/specs/2026-07-06-mpc-qp-continuous-bezier-trajectory-design.html`
  - `notes/todo.md`
  - `notes/todo/T302v-mpc-qp-safety-constrained-backend-plan.md`
