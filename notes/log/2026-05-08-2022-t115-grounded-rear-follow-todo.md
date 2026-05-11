# T115 Grounded Rear-Follow Todo Tree

## Meta

- Time: `2026-05-08 20:22 +0800`
- Stage: `together planner todo breakdown`
- Result: `pass`
- Todo: [T100/T115](../todo/T100-batched-together-planner-gpu-migration.md#t115-grounded-rear-follow-crossing-under-env_isaacsim)

## Purpose

- Convert the approved grounded rear-follow Isaac Lab design into executable todo leaves without creating a standalone implementation plan document.
- Record the dependency order and acceptance mapping for implementation under the new `env_isaacsim` headless runtime acceptance model.

## Todo Decisions

- The approved `T115` design is mapped into six execution leaves:
  - `T115a` grounded touchdown metrics and state-contract tightening
  - `T115b` touchdown / feet / base three-surface crossing validity
  - `T115c` deterministic grounded-phase fixture coverage
  - `T115d` `env_isaacsim` headless Isaac Lab runtime harness and diagnostics
  - `T115e` `env_isaacsim` headless runtime acceptance cases
  - `T115f` carry-forward union and final-code-state authority
- The decomposition follows the approved design hierarchy:
  - tighten the grounded phase contract first
  - then make crossing success depend on touchdown, foot path, and base path
  - then prove the planner-side grounded fixtures
  - then extend the runtime harness
  - then prove the real Isaac Lab headless runtime cases
  - finally close the carry-forward final union
- No standalone plan document was created; the todo tree lives only in repository memory.

## Dependency Structure

- `T115a` depends on completed `T114`, but has no upstream leaf dependency inside `T115`.
- `T115b` depends on `T115a`.
- `T115c` depends on `T115a` and `T115b`.
- `T115d` depends on `T115a` and `T115b`.
- `T115e` depends on `T115c` and `T115d`.
- `T115f` depends on `T115c` and `T115e`.

Recommended execution order:

1. `T115a`
2. `T115b`
3. `T115c`
4. `T115d`
5. `T115e`
6. `T115f`

## Acceptance Mapping

- `T115a` maps to:
  - grounded `front_cross`
  - grounded `rear_follow`
  - grounded-only `clear`
  - selected airborne rear-touchdown crossing outcome must fail
- `T115b` maps to:
  - touchdown not on `small`
  - foot path not colliding with `small`
  - base path not penetrating `small`
- `T115c` maps to:
  - `G1-G5` deterministic grounded-phase fixtures
  - `G3` rear-airborne invalidation
- `T115d` maps to:
  - runtime metric surfacing in headless Isaac Lab
  - reusable output-based diagnostics source
- `T115e` maps to:
  - `R1-R4` headless Isaac Lab acceptance cases under `env_isaacsim`
- `T115f` maps to:
  - `T113/T114` carry-forward obligations
  - final-code-state rerun-authority closure

## Verification

- Todo-only step; no implementation code changed in this log.
- Updated required repository memory surfaces:
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
- Execution leaves created: `6`
- New grounded-phase deterministic fixtures preserved from the design: `5`
  - `G1` through `G5`
- New runtime acceptance cases preserved from the design: `4`
  - `R1` through `R4`
- Design-specific three-surface crossing validation preserved:
  - touchdown surface
  - feet path surface
  - base path surface

## Conclusion

- The approved `T115` design is now converted into an executable todo tree and is ready for implementation.
- Recommended first execution leaf is `T115a`.

## Follow-up

- If implementation should begin, start with `T115a` and continue autonomously in dependency order.
- `T115d` and `T115e` should stay coupled to the real `env_isaacsim` headless runtime path rather than drifting back to viewer-image acceptance.

## Git Refs

- Baseline Ref: `current working tree after grounded rear-follow design approval`
- Candidate Ref: `working tree with T115 grounded rear-follow todo mapping and notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-08-grounded-rear-follow-env-isaacsim-design.md](../../docs/superpowers/specs/2026-05-08-grounded-rear-follow-env-isaacsim-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
