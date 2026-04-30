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

- T205: full-grid interactive viewer startup cost / manual semantic viewer confirmation

## Closed Children Archive

- T201: `semantic_raycaster` root traversal / static semantic merge / sensor tests landed; local tests pass
- T202: `extension/semantic_course.py` landed; local tests pass
- T203: semantic viewer env config landed; local tests pass
- T204: viewer semantic scanner path / diagnostics / marker partitioning landed; local tests pass
- T206: native semantic shape-pool landed for `semantic_course` + `semantic_raycaster`; local regression is green and compact runtime acceptance now requires `capsule` and `cone`
- T207: deterministic full-sub-terrain semantic layout and robust footprint grounding landed; targeted runtime small/large scan support landed

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

- Decide whether the full interactive viewer should keep the training-aligned full terrain grid or borrow the compact smoke strategy for startup practicality.
- Run one manual semantic viewer confirmation pass when interactive validation is needed.

## Node Details

### T207 deterministic full-sub-terrain semantic layout / footprint grounding

- status: done
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
  - spec review fixed seed/API defaults, tile-size resolution, targeted scan acceptance, exact grounding defaults, and fallback diagnostics
  - spec review passed after the above refinements; waiting on user spec approval before implementation planning
- evidence:
  - [2026-04-30-1508-semantic-course-random-layout-grounding-design.md](../log/2026-04-30-1508-semantic-course-random-layout-grounding-design.md)
  - [2026-04-30-1514-semantic-course-random-layout-spec-review.md](../log/2026-04-30-1514-semantic-course-random-layout-spec-review.md)
  - [2026-04-30-1518-semantic-course-random-layout-spec-review-approval.md](../log/2026-04-30-1518-semantic-course-random-layout-spec-review-approval.md)
  - [2026-04-30-1548-semantic-course-layout-grounding-implementation.md](../log/2026-04-30-1548-semantic-course-layout-grounding-implementation.md)
  - [2026-04-30-1619-semantic-course-random-layout-final-verification.md](../log/2026-04-30-1619-semantic-course-random-layout-final-verification.md)
- next:
  - no T207 action; continue T205 full-grid/manual viewer confirmation when visual validation is needed

#### T207 embedded implementation plan

This plan intentionally lives in branch memory instead of a separate `docs/superpowers/plans/` file per user request.

Control model:

- master-agent owns orchestration, context curation, cross-task integration, review gates, final verification, notes/log updates, and final reporting
- sub-agents own bounded implementation/test tasks with explicit file ownership
- sub-agents are not alone in the codebase; they must not revert others' edits and must adapt to already-landed changes
- implementation sub-agents should not run repository skill workflows unless explicitly asked; the delegating prompt is their task contract
- master-agent dispatches code tasks sequentially when write scopes overlap, and can dispatch review/test-only tasks in parallel after the implementation surface is stable
- after each implementation task, master-agent runs or delegates:
  - spec-compliance review against [../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md](../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md)
  - code-quality review for scope, API compatibility, deterministic behavior, and test adequacy

Task ownership:

1. **Layout API and deterministic anchors**
   - owner: sub-agent W1
   - write scope:
     - [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
     - [../../Go2Pvcnn/tests/test_semantic_course.py](../../Go2Pvcnn/tests/test_semantic_course.py)
   - responsibilities:
     - add exact defaults from the approved spec
     - add layout/grounding config structures or equivalent immutable defaults
     - implement `resolve_tile_size(...)`
     - replace fixed `_STAGE_LAYOUTS` coordinates with deterministic per-tile random layout generation
     - preserve `build_course_anchors(terrain_origins)` compatibility
     - expose `layout_fallback_used`
   - required tests:
     - same seed/layout is reproducible
     - different row/col layouts differ
     - bounds, center-safety, spacing, and canonical no-fallback checks pass
     - tight synthetic fallback still respects margin and center safety

2. **Footprint grounding**
   - owner: sub-agent W2 after W1 lands, or same worker if master-agent decides coupling is too high
   - write scope:
     - [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
     - [../../Go2Pvcnn/tests/test_semantic_course.py](../../Go2Pvcnn/tests/test_semantic_course.py)
   - responsibilities:
     - generate shape-aware footprint sample xy points
     - update pure `ground_course_anchors(...)` to use multi-point terrain sampling
     - update runtime grounding to batch all footprint samples through the existing terrain raycast sampler
     - default to max finite footprint height and `0.015m` embed depth
   - required tests:
     - footprint samples exist for each native shape kind
     - center-only fake sampler no longer controls uneven-terrain result
     - `center_z = max_footprint_z - 0.015 + bottom_to_center_offset(shape)` for default config

3. **Targeted runtime scan support**
   - owner: sub-agent W3 after W1 API is stable
   - write scope:
     - [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
     - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
     - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
   - responsibilities:
     - add/select helper for locating a generated anchor by stage/class
     - add test fixture support to scan near a selected anchor instead of assuming tile center visibility
     - keep production viewer behavior unchanged unless a tiny shared helper is clearly useful
   - required tests:
     - targeted S4 small scan reports semantic id `1`
     - targeted S4 large scan reports semantic id `2`
     - test completes within bounded warmup/update steps

4. **Integration verification and notes**
   - owner: master-agent, with optional test-only sub-agent after implementation tasks land
   - write scope:
     - [../log/index.md](../log/index.md)
     - this branch page
     - [../todo.md](../todo.md)
     - per-verification logs under [../log/](../log/)
   - responsibilities:
     - run focused local tests first
     - run semantic viewer/runtime subset in `env_isaaclab` when local tests pass
     - record exact commands, metrics, failures, and follow-up nodes
     - decide whether any failure is implementation, spec, environment, or runtime-resource related

Parallelization policy:

- W1 and W2 both touch `semantic_course.py`, so they should not edit concurrently unless W1 is split into a read-only exploration task and W2 waits for W1's merged API.
- W3 can begin as a read-only exploration of runtime fixture requirements while W1 is implementing, but W3 must not write until W1's public anchor API is stable.
- Spec-compliance and code-quality reviews for a completed task can run in parallel because they are read-only.
- Runtime smoke testing can run in parallel with notes summarization only after all code changes are stable.

Acceptance metrics:

- `pytest Go2Pvcnn/tests/test_semantic_course.py -q` passes.
- `pytest Go2Pvcnn/tests/test_semantic_raycaster.py -q` passes or any failure is documented as unrelated.
- `pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q` passes in `/home/lhy/anaconda3/envs/env_isaaclab` or runtime-resource caveat is logged with evidence.
- Pure tests prove:
  - no fixed center-only `_STAGE_LAYOUTS` anchor table remains as the layout source
  - default S4 anchors spread outside the old `1.5m` center scanner footprint
  - same seed is reproducible and different tiles differ
  - default grounding uses footprint max height minus `0.015m` plus shape offset
- Runtime/semantic tests prove:
  - targeted small and large semantic scans report class ids `1` and `2`
  - native shape-pool coverage still includes `capsule` and `cone`

Master-agent review checklist:

- confirm no training config migration was introduced
- confirm no viewer-owned layout logic was added
- confirm no random global state or Python process-randomized `hash()` controls layout
- confirm existing no-argument callers remain deterministic
- confirm fallback does not silently mask canonical layout failures
- confirm notes/log/todo are aligned before claiming completion

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
