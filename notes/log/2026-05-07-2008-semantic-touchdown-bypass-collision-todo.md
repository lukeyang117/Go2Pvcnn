# Semantic Touchdown/Bypass/Collision Todo Tree

## Meta

- Time: `2026-05-07 20:08 +0800`
- Stage: `together planner todo breakdown`
- Result: `pass`
- Todo: [T100/T113](../todo/T100-batched-together-planner-gpu-migration.md#t113-semantic-touchdown-bypass--collision-redesign)

## Purpose

- Convert the approved semantic touchdown/bypass/collision redesign into executable todo leaves without creating a standalone implementation plan document.
- Record the execution ordering, dependency edges, and acceptance mapping for later implementation approval.

## Todo Decisions

- The approved design is now mapped into five execution leaves under `T113`:
  - `T113a` semantic-valid support query and touchdown/support consistency
  - `T113b` always-on `K=3` foothold-policy candidate axis
  - `T113c` small-cross preference and large-bypass foothold policy
  - `T113d` height-aware swing and continuous collision model
  - `T113e` deterministic tests, metrics, and diagnostics surfacing
- The breakdown intentionally follows the design's responsibility boundaries:
  - terrain/legal-support first
  - candidate-axis policy second
  - obstacle-class foothold behavior third
  - swing/collision validation fourth
  - explicit metrics/tests last, after behavior is in place
- No standalone plan document was created; the todo tree lives in repository memory only, per workflow contract.

## Dependency Structure

- `T113a` has no implementation dependency inside `T113`.
- `T113b` depends on `T113a`.
- `T113c` depends on `T113a` and `T113b`.
- `T113d` depends on `T113b` and `T113c`.
- `T113e` depends on `T113a`, `T113b`, `T113c`, and `T113d`.

Recommended execution order:

1. `T113a`
2. `T113b`
3. `T113c`
4. `T113d`
5. `T113e`

## Acceptance Mapping

- `T113a` maps to:
  - legal-support filtering
  - `small/large` not valid support surfaces
  - `support_xy_z_consistency`
- `T113b` maps to:
  - fixed `K=3` in all scenes
  - no center-default privilege in obstacle-free terrain
- `T113c` maps to:
  - `small` as non-step-on but not mandatory behind-obstacle rule
  - `large` avoidance starting from foothold/touchdown policy
- `T113d` maps to:
  - height-aware swing clearance
  - body/thigh/calf collision coverage
  - soft penalty plus hard infeasible behavior
- `T113e` maps to:
  - fixtures `F1` through `F9`
  - explicit metric assertions
  - final traceability from spec acceptance indicators to tests

## Verification

- Todo-only step; no implementation code changed in this log.
- Updated all required repository memory surfaces:
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
- Execution leaves created: `5`
- Explicit metric names preserved from the design: `11`
- Dependency edges recorded: `9`

## Conclusion

- The approved design has been converted into an executable todo tree and is ready for later implementation approval leaf by leaf.
- The next execution step is to choose which `T113` leaf to implement first; the recommended start is `T113a`.

## Follow-up

- If the user wants implementation to begin, approve a specific leaf under `T113`.
- Recommended first leaf: `T113a`, then continue in dependency order unless a strong reason emerges to regroup write scopes.

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 (2026-05-07 20:08 +0800); T113 todo mapping + notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md](../../docs/superpowers/specs/2026-05-07-semantic-touchdown-bypass-collision-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
