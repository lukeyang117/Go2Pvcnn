# T116 Implementation Plan And Subagent Dispatch

## Purpose

Record the executable T116 plan split and the orchestration rules before code implementation begins.

## Stage

Todo breakdown / execution control for `Go2Pvcnn/extension/batched_together_planner`.

## Related Todo

- [T100/T116](../todo/T100-batched-together-planner-gpu-migration.md#t116-k5-mode-first-small-obstacle-crossing-rewrite)
- [T116a](../todo/T100-batched-together-planner-gpu-migration.md#t116a-contractschemahorizonk5-rewrite)

## Procedure

- Read repository constraints, T116 spec, T100 branch page, planner notes, and current dashboard.
- Integrated Fermat subagent boundary review into the T116 todo tree.
- Closed the review subagent after capturing its findings.
- Updated dashboard and log index so T116a is the next executable leaf.
- Marked T116a `doing` and dispatched implementation worker James (`019e0c25-1894-7c30-acd0-7882588adce1`).

## Key Decisions

- Use T116a as an interface-first leaf: config constants, result schema, direct 50-step consumers, and the minimal `planner.py` candidate-axis compatibility needed to avoid the old `semantic_candidate_count == 3` gate.
- Run T116b through T116e sequentially because `planner.py`, `parameterization.py`, `schedule.py`, and `costs.py` are tightly coupled.
- Keep T116f test cleanup under main-agent serialized ownership or incremental leaf-by-leaf updates, not as a parallel worker against the same large test files.
- Require TDD red/green for each implementation leaf and require spec plus quality review before moving to the next leaf.

## Result

Plan split is active and T116a implementation is in progress under a worker subagent.

## Follow-Up

- Poll subagents until complete.
- After T116a returns, run spec compliance review, then code quality review.
- Update this branch and create implementation verification logs after T116a red/green and review.

## Git Refs

- Baseline Ref: working tree on top of `7cf6c11`
- Candidate Ref: uncommitted notes/spec/test/planner workspace
- Key Files:
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
  - [../../docs/superpowers/specs/2026-05-09-k5-mode-first-cross-small-design.md](../../docs/superpowers/specs/2026-05-09-k5-mode-first-cross-small-design.md)
