# T200 Semantic Static Course Viewer

## Current State

- Viewer-first semantic static-course path is implemented and verified in local tests.
- `semantic_height_scanner` is the active scanner; compact `env_isaaclab` headless smoke passes on default `together`.
- Viewer colors are now:
  - terrain white
  - small obstacle green
  - large obstacle red
- Native shape-pool expansion is also landed:
  - `sphere`
  - `cuboid`
  - `cylinder`
  - `capsule`
  - `cone`
- `small` and `large` share the shape pool; slot shape choice is deterministic per `(stage, row, col, slot, semantic_class)`.
- Compact runtime acceptance now explicitly requires both `capsule` and `cone`.
- New design follow-up `T207` is active: replace center-clustered semantic anchors with full sub-terrain deterministic random layouts and footprint-based terrain grounding.
- Remaining existing follow-up `T205`: full-grid interactive startup cost and one manual viewer confirmation.

## Open Children

- T207: deterministic full-sub-terrain semantic layout and robust footprint grounding
- T205: full-grid interactive viewer startup cost / manual semantic viewer confirmation

## Closed Children Archive

- T201: `semantic_raycaster` root traversal / static semantic merge / sensor tests landed; local tests pass
- T202: `extension/semantic_course.py` landed; local tests pass
- T203: semantic viewer env config landed; local tests pass
- T204: viewer semantic scanner path / diagnostics / marker partitioning landed; local tests pass
- T206: native semantic shape-pool landed for `semantic_course` + `semantic_raycaster`; local regression is green and compact runtime acceptance now requires `capsule` and `cone`

## Related Logs

- [2026-04-29-2209-semantic-static-course-viewer-design.md](../log/2026-04-29-2209-semantic-static-course-viewer-design.md)
- [2026-04-29-2234-semantic-static-course-viewer-spec-review.md](../log/2026-04-29-2234-semantic-static-course-viewer-spec-review.md)
- [2026-04-29-2318-semantic-static-course-parallel-review-convergence.md](../log/2026-04-29-2318-semantic-static-course-parallel-review-convergence.md)
- [2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md](../log/2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md)
- [2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md](../log/2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md)
- [2026-04-30-0215-semantic-viewer-empty-marker-fix.md](../log/2026-04-30-0215-semantic-viewer-empty-marker-fix.md)
- [2026-04-30-1343-semantic-native-shape-pool-design.md](../log/2026-04-30-1343-semantic-native-shape-pool-design.md)
- [2026-04-30-1351-semantic-native-shape-pool-spec-review.md](../log/2026-04-30-1351-semantic-native-shape-pool-spec-review.md)
- [2026-04-30-1432-semantic-native-shape-pool-compact-runtime-acceptance.md](../log/2026-04-30-1432-semantic-native-shape-pool-compact-runtime-acceptance.md)
- [2026-04-30-1508-semantic-course-random-layout-grounding-design.md](../log/2026-04-30-1508-semantic-course-random-layout-grounding-design.md)

## Git Refs

- Last Feature Commit: `7bb89ed`
- Last Verified Commit: `7bb89ed`
- Current Work Ref: `working tree on top of 7bb89ed (2026-04-30 14:32 +0800); waiting on full-grid/manual viewer confirmation`
- Key Files:
  - [../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md](../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md)
  - [../../docs/superpowers/specs/2026-04-30-semantic-native-shape-pool-design.md](../../docs/superpowers/specs/2026-04-30-semantic-native-shape-pool-design.md)
  - [../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md](../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md)
  - [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py](../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py)
  - [../../Go2Pvcnn/tests/test_semantic_course.py](../../Go2Pvcnn/tests/test_semantic_course.py)
  - [../../Go2Pvcnn/tests/test_semantic_raycaster.py](../../Go2Pvcnn/tests/test_semantic_raycaster.py)

## Next Step

- For `T207`, create the implementation plan for deterministic full-tile layout sampling, footprint terrain grounding, and targeted semantic-hit runtime tests.
- Decide whether the full interactive viewer should keep the training-aligned full terrain grid or borrow the compact smoke strategy for startup practicality.
- Run one manual semantic viewer confirmation pass when interactive validation is needed.

## Node Details

### T207 deterministic full-sub-terrain semantic layout / footprint grounding

- why-created:
  - user observed semantic objects are clustered near the center plane of each sub-terrain
  - current `_STAGE_LAYOUTS` only spans roughly the central scanner window even though active sub-terrains are `8m x 8m`
  - user explicitly chose full sub-terrain spread, deterministic per-tile randomness, and upright robust grounding
- approved-design:
  - replace fixed anchor coordinates with deterministic per-tile pseudo-random layout generation
  - keep S1-S4 object counts and semantic classes unchanged
  - sample local xy across most of each tile with margin, center safety, and minimum spacing
  - ground objects with footprint multi-point terrain samples plus a small embedding depth
  - no default guarantee that env0's initial `1.5m` scanner window sees all semantic objects
- evidence:
  - [2026-04-30-1508-semantic-course-random-layout-grounding-design.md](../log/2026-04-30-1508-semantic-course-random-layout-grounding-design.md)
- next:
  - write implementation plan after spec review and user spec approval

### T205 full-grid interactive viewer startup / manual confirmation

- why-active:
  - compact headless runtime and native shape-pool coverage are proven
  - full-grid interactive startup cost is still unmeasured after the recent semantic expansion
  - manual visual confirmation is still the fastest way to judge whether the richer shape pool looks good in practice
- likely decisions:
  - keep full training-aligned grid for interactive viewer
  - or adopt compact terrain-grid setup as the default smoke/diagnostic path and reserve full-grid for explicit manual checks
- keep under direct review:
  - interactive startup time
  - visual distribution of `sphere/cuboid/cylinder/capsule/cone`
  - whether compact runtime smoke is enough as the standing automated acceptance proof
