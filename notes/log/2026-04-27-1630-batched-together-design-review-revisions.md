# Batched Together Design Review Revisions

## Meta

- Time: `2026-04-27 16:30 +0800`
- Stage: `batched_together_planner design review`
- Result: `issues incorporated`
- Todo: [T100](../todo/T100-batched-together-planner-gpu-migration.md)

## Review Sources

Four read-only subagents reviewed the design:

- training-chain parallelism and GPU purity
- raw planner semantic/tensor alignment
- manager/reward/cache contract alignment
- test richness and static guardrail coverage

All four returned `ISSUES`.

## Main Revisions

- Added a hard rule that together training must not branch on GPU masks before planner call.
- Added a GPU cache ABI requirement and marked the existing CPU-canonical converter as legacy/reference-only for together training.
- Added manager-owned phase/current-reference as the reward frame source; `episode_length_buf % horizon` is not compatible with together accept-mask semantics.
- Added a `TrajectoryManagerProtocol` surface: `refresh_from_env`, `current_reference`, `current_frame_ids`, `reset_envs`, `_trajectory_reference_cache`, and same-step idempotence.
- Added backend factory requirements for train/play/viewer attach paths.
- Added safe fallback semantics:
  - `safe_fallback` is an explicit together result field;
  - `fallback_mask` rows receive GPU standstill/rehome cache;
  - first/reset/cache-invalid rows must not accept invalid planner rows.
- Expanded raw semantic contract to include config, schedule, seeds/CEM, parameterization, support query, costs, IK diagnostics, fallback/rehome, and result schema.
- Expanded parity tests into a matrix with fields, tolerances, scenarios, and diagnostic sources.
- Expanded static guardrail into an AST/text scan design with scan manifest, exclusions, forbidden calls, dynamic sub-batch detection, and `for`-loop restrictions.

## Unresolved Spec Point

The formal spec must still choose the exact host-scalar planner attempt cadence:

- unconditional full-`N` call on each manager advance; or
- fixed host-cadence full-`N` attempts plus GPU fallback rows between attempts.

The forbidden behavior is already clear: planner execution cannot be gated by `torch.any(replan_mask)` or any GPU mask converted to Python.

## Verification

- Notes-only update.
- No implementation code changed.
- `T100` branch page and dashboard now include review findings.

## Git Refs

- Baseline Ref: `working tree on top of 7cf6c11 after 16:22 design note`
- Candidate Ref: `working tree on top of 7cf6c11 (2026-04-27 16:30 +0800); design review revisions added`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
