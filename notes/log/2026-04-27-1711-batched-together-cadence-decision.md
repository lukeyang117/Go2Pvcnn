# Batched Together Cadence Decision

## Meta

- Time: `2026-04-27 17:11 +0800`
- Stage: `batched_together_planner runtime design`
- Result: `design decision recorded`
- Todo: [T100](../todo/T100-batched-together-planner-gpu-migration.md)

## Decision

The together backend should update trajectory only when:

- velocity command changes;
- reset occurs;
- the `0.7s` / `35` frame replan interval is reached.

When a planner attempt happens, it is still a full-`N` planner call. The new backend must not use dynamic sub-batches.

## Important Constraint

The planner attempt trigger must be host-visible:

- acceptable: a command dirty token, command version counter, explicit command manager hook, reset hook, or host step counter;
- forbidden: comparing GPU command tensors and branching on `torch.any(command_changed_mask)` or similar GPU reductions.

Per-env tensor masks may still decide which cache rows accept the new full-batch result, using `torch.where`.

## Design Impact

- `TogetherTrajectoryManager` needs an explicit command/reset/interval trigger contract.
- `replan_mask [N]` stays tensor-only and controls row acceptance.
- If one env command changes, the full planner receives all env rows; unchanged rows normally keep their old cache through `keep_mask`.
- If no host trigger is pending and the interval has not expired, manager should not run the planner and should only advance/read the existing cache phase as appropriate.

## Verification

- Notes-only update.
- No implementation code changed.

## Git Refs

- Baseline Ref: `working tree on top of 7cf6c11 after T100 review revisions`
- Candidate Ref: `working tree on top of 7cf6c11 (2026-04-27 17:11 +0800); cadence decision added`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
