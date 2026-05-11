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
- Follow-up `T207` is complete: full sub-terrain deterministic random layouts, footprint-based terrain grounding, and targeted runtime small/large scans are landed.
- Remaining existing follow-up `T205`: full-grid interactive startup cost and one manual viewer confirmation.

## Open Children

- T208: small obstacle height reduction follow-up
- T205: full-grid interactive viewer startup cost / manual semantic viewer confirmation

## Closed Children Archive

- T201: `semantic_raycaster` root traversal / static semantic merge / sensor tests landed; local tests pass
- T202: `extension/semantic_course.py` landed; local tests pass
- T203: semantic viewer env config landed; local tests pass
- T204: viewer semantic scanner path / diagnostics / marker partitioning landed; local tests pass
- T206: native semantic shape-pool landed for `semantic_course` + `semantic_raycaster`; local regression is green and compact runtime acceptance now requires `capsule` and `cone`
- T207: deterministic full-sub-terrain semantic layout and robust footprint grounding landed; targeted runtime small/large scan support landed

## Related Logs

- [2026-05-07-2044-small-obstacle-height-follow-up-todo.md](../log/2026-05-07-2044-small-obstacle-height-follow-up-todo.md)
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
- [2026-04-30-1514-semantic-course-random-layout-spec-review.md](../log/2026-04-30-1514-semantic-course-random-layout-spec-review.md)
- [2026-04-30-1518-semantic-course-random-layout-spec-review-approval.md](../log/2026-04-30-1518-semantic-course-random-layout-spec-review-approval.md)
- [2026-04-30-1548-semantic-course-layout-grounding-implementation.md](../log/2026-04-30-1548-semantic-course-layout-grounding-implementation.md)
- [2026-04-30-1619-semantic-course-random-layout-final-verification.md](../log/2026-04-30-1619-semantic-course-random-layout-final-verification.md)

## Git Refs

- Last Feature Commit: `130c635`
- Last Verified Commit: `130c635`
- Current Work Ref: `working tree on top of 130c635 (2026-04-30 16:19 +0800); T207 complete; unrelated raw/NvStreamer dirty entries present`
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

- Decide whether `T208` should run before or after the current `T113` planner-semantics leaves, depending on whether obstacle geometry is still masking planner behavior in viewer validation.
- Decide whether the full interactive viewer should keep the training-aligned full terrain grid or borrow the compact smoke strategy for startup practicality.
- Run one manual semantic viewer confirmation pass when interactive validation is needed.

## Node Details

### Historical Summary

- `T207`: full-sub-terrain deterministic semantic layout, footprint grounding, and targeted runtime scans are complete. The old embedded implementation plan remains historical context only.
- `T205`: full-grid interactive startup remains a manual/background viewer concern, not an active mainline issue.
- `T208`: small obstacle height reduction is complete and serves only as background geometry evidence for later planner work.

### Keep In Mind

- `T200` is a completed support branch, not an active front.
- If future work reopens semantic viewer/runtime behavior, create a fresh child node instead of restoring the long historical implementation narrative here.
