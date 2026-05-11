# T114 State-Machine Todo Tree

## Meta

- Time: `2026-05-08 13:21 +0800`
- Stage: `together planner todo breakdown`
- Result: `pass`
- Todo: [T100/T114](../todo/T100-batched-together-planner-gpu-migration.md#t114-state-machine-touchdown-front-end-redesign)

## Purpose

- Convert the approved state-machine touchdown-front-end design into executable todo leaves without creating a standalone implementation plan document.
- Record the dependency order and acceptance mapping for later implementation approval.

## Todo Decisions

- The approved `T114` design is now mapped into seven execution leaves:
  - `T114a` state classification masks and corridor summaries
  - `T114b` candidate action-segment representation
  - `T114c` touchdown boundary-margin controls
  - `T114d` four-leg consistency and whole-body posture scoring
  - `T114e` anchor-to-touchdown foot/leg path clearance
  - `T114f` state-aware invalidation and selection rules
  - `T114g` deterministic metric/fixture traceability and final rerun authority
- The decomposition follows the approved design hierarchy:
  - classify state first
  - represent candidates as action segments
  - then add boundary, consistency, and path constraints
  - then add state-aware invalidation/selection
  - finally close metric/fixture/rerun-authority coverage
- No standalone plan document was created; the todo tree lives only in repository memory.

## Dependency Structure

- `T114a` has no upstream leaf dependency inside `T114`.
- `T114b` depends on `T114a`.
- `T114c` depends on `T114a` and `T114b`.
- `T114d` depends on `T114a`, `T114b`, and `T114c`.
- `T114e` depends on `T114a`, `T114b`, `T114c`, and `T114d`.
- `T114f` depends on `T114a`, `T114b`, `T114c`, `T114d`, and `T114e`.
- `T114g` depends on all prior `T114` leaves.

Recommended execution order:

1. `T114a`
2. `T114b`
3. `T114c`
4. `T114d`
5. `T114e`
6. `T114f`
7. `T114g`

## Acceptance Mapping

- `T114a` maps to:
  - unified state framework in small and no-small scenes
  - explicit `cruise / approach / ready_to_cross / front_cross / rear_follow / bypass / clear`
- `T114b` maps to:
  - candidate-as-action-segment semantics
  - state/path/consistency/posture diagnostics presence
- `T114c` maps to:
  - touchdown distance margin from `small`
  - near-boundary penalty/invalidation
- `T114d` maps to:
  - front-pair consistency
  - rear-pair follow consistency
  - whole-body posture quality
- `T114e` maps to:
  - anchor-to-touchdown foot clearance
  - anchor-to-touchdown leg clearance
  - candidate path collision flag
- `T114f` maps to:
  - state-aware `approach / cross / bypass` selection
  - large bypass/refusal
  - deterministic state transitions
- `T114g` maps to:
  - `F1`, `F1b`, `F2`-`F15` test coverage
  - explicit metric assertions
  - final-code-state rerun-authority rule

## Verification

- Todo-only step; no implementation code changed in this log.
- Updated all required repository memory surfaces:
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
- Execution leaves created: `7`
- Deterministic fixtures preserved from the design: `16`
  - `F1`, `F1b`, and `F2` through `F15`
- Explicit metrics preserved from the design: `21`

## Conclusion

- The approved `T114` design is now converted into an executable todo tree and is ready for leaf-by-leaf implementation approval.
- Recommended first execution leaf is `T114a`.

## Follow-up

- If the user wants implementation to begin, approve a specific `T114` leaf.
- Recommended start: `T114a`, then continue in dependency order unless write scopes are regrouped intentionally.

## Git Refs

- Baseline Ref: `current working tree after T114 design-state sync`
- Candidate Ref: `working tree with T114 todo mapping and notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md](../../docs/superpowers/specs/2026-05-08-semantic-touchdown-state-machine-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
