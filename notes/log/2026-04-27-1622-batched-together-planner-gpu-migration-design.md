# Batched Together Planner GPU Migration Design

## Meta

- Time: `2026-04-27 16:22 +0800`
- Stage: `batched_together_planner design`
- Result: `design recorded`
- Todo: [T100](../todo/T100-batched-together-planner-gpu-migration.md)

## Purpose

- Record the user-approved design direction before implementation.
- Preserve the strict training-path constraints for future agents:
  - no CPU packages or CPU synchronization in the training path;
  - no dynamic sub-batch replanning;
  - no hot-path Python loops over env, batch, horizon, candidate, or leg dimensions.

## Design Decisions

- Use a new native IsaacLab GPU backend in `Go2Pvcnn/extension/batched_together_planner/`.
- Keep `extension.batched_planner` as a legacy rollback backend.
- Add `planner_backend = "together" | "legacy"`; default should be `together`.
- Strict new-backend timing contract: `35` frames, `dt = 0.02`, `0.7s`.
- Use fixed-shape full-batch planning:
  - planner always consumes all envs `[N, ...]`;
  - `replan_mask [N]` decides per-env cache replacement, not planner batch shape;
  - manager blends old and new cache rows with `torch.where`.
- Require both:
  - A-level behavior alignment with raw planner effects;
  - B-level tensor parity on core planner outputs.
- Add a dedicated static guardrail child to detect forbidden training-path code patterns.

## Manager Old/New Cache Rule

The new manager keeps:

- `old_cache`
- `new_cache`
- `phase_counter`
- `replan_mask`
- `new_ok_mask`
- `accept_mask`

The accepted row set is:

```text
must_replace_mask = first_cache OR reset_mask OR cache_invalid
soft_replan_mask = replan_mask AND NOT must_replace_mask
new_ok_mask = result.feasible OR result.has_safe_fallback
accept_mask = must_replace_mask OR (soft_replan_mask AND new_ok_mask)
```

Every cache tensor is updated with tensor blending instead of dynamic row writes.
Phases reset to `0` for accepted rows and otherwise advance/clamp.

## Review Requested

The user requested subagent design review for:

- training-chain parallelism and GPU purity;
- raw planner alignment;
- current manager/reward/cache alignment;
- test richness and guardrail coverage.

## Verification

- Design only; no code was changed.
- Todo dashboard, T100 branch page, and log index are updated.
- Formal spec and implementation plan are still pending.

## Git Refs

- Baseline Ref: `7cf6c11`
- Candidate Ref: `working tree on top of 7cf6c11 (2026-04-27 16:22 +0800); notes-only design update`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
