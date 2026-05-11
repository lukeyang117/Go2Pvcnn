# T116i Nonzero Speed And Hard-Reason Todo

## Purpose

Record the approved todo split for the T116 follow-up that removes nonzero-command `beta=0` candidates and adds hard-constraint reason diagnostics.

## Stage

T100 / T116 batched together planner runtime and viewer diagnostics.

## Related Todo

- [T100 / T116i](../todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics)

## Procedure

- Reviewed the user-approved design:
  - [../../docs/superpowers/specs/2026-05-10-nonzero-speed-hard-reason-design.md](../../docs/superpowers/specs/2026-05-10-nonzero-speed-hard-reason-design.md)
- Added `T116i` as a child of T116, not a new root.
- Updated the root dashboard active fronts/open leaves.
- Updated the T100 branch page with scope, constraints, acceptance tests, and dependencies.

## Key Requirements Captured

- Nonzero command candidate tables must not include `beta=0`.
- Zero command may still produce hold/standstill through the existing command hold path.
- All-hard candidate cases must use fixed-shape hard-reason masks and rank costs instead of relying only on a flat barrier.
- Infeasible viewer/headless output must explain hard reasons.
- Tests must cover flat/no-semantic directions, flat small-obstacle crossing directions, deterministic all-hard ranking, and terminal hard-reason output.
- Runtime tests must use `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`, headless output, and timeout cleanup.

## Result

Todo recorded. No production code changed in this pass.

## Follow-Up

Create an implementation plan for `T116i`, then update the together planner and authority tests in place.

## Git Refs

- Baseline Ref: `979b2b5`
- Candidate Ref: working tree
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [../../docs/superpowers/specs/2026-05-10-nonzero-speed-hard-reason-design.md](../../docs/superpowers/specs/2026-05-10-nonzero-speed-hard-reason-design.md)
