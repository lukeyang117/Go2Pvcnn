# Outdated Non-mpc_qp Roots Archive

Created: 2026-07-06

This page records the user decision to treat all non-`mpc_qp` work as outdated context. The original branch pages and per-test logs are not deleted; they remain available for archaeology or explicit reopening.

## Archive Rule

- Current/default working front: [../T302v-mpc-qp-safety-constrained-backend-plan.md](../T302v-mpc-qp-safety-constrained-backend-plan.md)
- Reopen any item below only if the user explicitly asks for it.
- Do not continue old RL reward/curriculum/policy-eval lines just because they appear in historical logs.
- Do not route new `mpc_qp` safety work into old `T300`/`T302k` dense-MPC branches.

## Archived Root Summary

| Root | Old Stage | Archive Status | Where To Read If Reopened |
| --- | --- | --- | --- |
| T100 | batched together planner migration | historical context | [../T100-batched-together-planner-gpu-migration.md](../T100-batched-together-planner-gpu-migration.md) |
| T200 | semantic static course viewer | done/context | [../T200-semantic-static-course-viewer.md](../T200-semantic-static-course-viewer.md) |
| T300/T300e | unified dense `planner_backend="mpc"` backend | superseded for current work by opt-in `mpc_qp`; keep as historical default-MPC context | [../T300-unified-dense-mpc-backend.md](../T300-unified-dense-mpc-backend.md), [../T300e-mpc-continuous-swing-window-plan.md](../T300e-mpc-continuous-swing-window-plan.md) |
| T301 | viewer reset/step mode | context only | [../T301-viewer-r-key-grounded-reset.md](../T301-viewer-r-key-grounded-reset.md) |
| T302 | body/leg collision baseline | historical context for semantic safety | [../T302-mpc-body-leg-height-field-collision-safety.md](../T302-mpc-body-leg-height-field-collision-safety.md) |
| T302g-T302u | MPC semantic RL, command-frame, low-small reward/curriculum, policy eval, map contact | outdated context; many were diagnostics or reward-training attempts before `mpc_qp` | branch pages under [../](../) |
| T302k | parametric MPC trajectory contract / low-small loss redesign | historical default-MPC trajectory work; do not tune unless explicitly reopened | [../T302k-parametric-mpc-trajectory-contract.md](../T302k-parametric-mpc-trajectory-contract.md), [../T302k-low-small-loss-redesign-plan.md](../T302k-low-small-loss-redesign-plan.md) |
| T302s/T302r/T302q | flat-small RL reward/curriculum attempts | outdated training line; current priority is QP safety backend, not reward continuation | [../T302s-env-level-collision-curriculum-plan.md](../T302s-env-level-collision-curriculum-plan.md), [../T302r-go2-geometry-clearance-reward-plan.md](../T302r-go2-geometry-clearance-reward-plan.md), [../T302q-flat-small-avoidance-reward-plan.md](../T302q-flat-small-avoidance-reward-plan.md) |

## Current Replacement Focus

[../T302v-mpc-qp-safety-constrained-backend-plan.md](../T302v-mpc-qp-safety-constrained-backend-plan.md) is the active memory for:

- isolated `planner_backend="mpc_qp"`
- `runtime.qp_iterations`
- semantic touchdown keepout
- high height-variation step cap
- low-small swing/contact repair
- FK body/leg/root safety diagnostics
- viewer strict crossing probes
- 1024/1024 GPU smoke evidence
- viewer `--qp-iterations` CLI compatibility

## Evidence Pointers

- [../../log/2026-07-06-mpc-qp-viewer-qp-iterations-cli-fix.md](../../log/2026-07-06-mpc-qp-viewer-qp-iterations-cli-fix.md)
- [../../log/2026-07-03-mpc-qp-strict-contact-crossing-final.md](../../log/2026-07-03-mpc-qp-strict-contact-crossing-final.md)
- [../../log/2026-07-03-human12-mpc-qp-command-update.md](../../log/2026-07-03-human12-mpc-qp-command-update.md)
