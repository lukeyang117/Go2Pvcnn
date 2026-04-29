# T200 Semantic Static Course Viewer

## Current State

- User-approved design direction is recorded for a viewer-first semantic static obstacle course tied to terrain difficulty.
- Training config remains unchanged for now; the immediate target is a derived viewer config that can later be migrated into `teacher_elevation_trajectory_env_cfg.py`.
- The old inherited `height_scanner` must be removed in the viewer config and replaced by `semantic_height_scanner`.
- `semantic_raycaster` is in scope for redesign and tests, not just reuse.
- Source inspection established that `startup` is too late for semantic prop spawning because sensor warp-mesh initialization happens on `sim.reset()` / timeline `PLAY` before `startup`. Semantic props must exist by `prestartup`.
- The written spec has passed subagent review with no blocking issues; advisory refinements were folded back into the spec.
- Parallel technical/detail/completeness review found and resolved additional blockers in the spec:
  - semantic viewer scene must be non-replicated
  - semantic-course root containers must always exist
  - raster size and diagnostics contracts are now explicit
  - semantic rollout success is required on default `together`
- Implementation slices for sensor, course/config, and viewer integration are now landed in the working tree.
- Local unit/static/viewer tests are passing.
- Compact real `env_isaaclab` headless runtime smoke now passes for:
  - `semantic_height_scanner_contract`
  - default `together` semantic smoke
- The remaining acceptance gap is no longer semantic correctness in compact smoke; it is full-grid interactive startup cost and manual viewer confirmation.

## Open Children

- T205: full-grid interactive viewer startup cost / manual semantic viewer confirmation

## Closed Children Archive

- T201: `semantic_raycaster` root traversal / static semantic merge / sensor tests landed; local tests pass
- T202: `extension/semantic_course.py` landed; local tests pass
- T203: semantic viewer env config landed; local tests pass
- T204: viewer semantic scanner path / diagnostics / marker partitioning landed; local tests pass

## Related Logs

- [2026-04-29-2209-semantic-static-course-viewer-design.md](../log/2026-04-29-2209-semantic-static-course-viewer-design.md)
- [2026-04-29-2234-semantic-static-course-viewer-spec-review.md](../log/2026-04-29-2234-semantic-static-course-viewer-spec-review.md)
- [2026-04-29-2318-semantic-static-course-parallel-review-convergence.md](../log/2026-04-29-2318-semantic-static-course-parallel-review-convergence.md)
- [2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md](../log/2026-04-29-2348-semantic-static-course-implementation-and-local-verification.md)
- [2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md](../log/2026-04-29-2359-semantic-static-course-env-isaaclab-compact-runtime-smoke.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `not yet applicable; design-only task`
- Current Work Ref: `working tree on top of 6279bc4 (2026-04-29 22:09 +0800); spec + notes design update`
- Key Files:
  - [../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md](../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py](../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py)
  - [../../Go2Pvcnn/extension](../../Go2Pvcnn/extension)

## Next Step

- Decide whether the full interactive viewer should keep the training-aligned full terrain grid or borrow the compact smoke strategy for startup practicality.
- Run one manual semantic viewer confirmation pass when interactive validation is needed.

## Execution Model

### Main Agent Responsibilities

- own the design source of truth and decision changes
- own sequencing, integration order, and conflict resolution
- review subagent findings for architectural correctness and scope discipline
- assign implementation slices with disjoint write scopes
- review returned code and verification before accepting it

### Parallel Review Subagents

- `R1 semantic_raycaster review`
  - check recursive root traversal, static semantic merge behavior, data contract, and likely edge cases
- `R2 semantic_course review`
  - check tile-based generation, `prestartup` timing, grounded placement, and terrain-difficulty coupling
- `R3 metrics/viewer review`
  - check viewer consumption contract, semantic hit diagnostics, and test/metric completeness

### Implementation Worker Direction

- workers own code changes and focused tests in clearly separated write scopes
- workers return:
  - what changed
  - what tests they ran
  - unresolved concerns or interface assumptions
- main agent does not delegate the final acceptance decision

### Intended Write-Scope Split

- `W1 sensor/tests`
  - `Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/*`
  - sensor-focused tests
- `W2 course/config/tests`
  - `Go2Pvcnn/extension/semantic_course.py`
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_semantic_viewer_env_cfg.py`
  - config/course-focused tests
- `W3 viewer/tests`
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - viewer integration tests

This split is provisional and can be adjusted by the main agent after the review subagents report back.

## Node Details

### Why Created

- The user wants a semantic obstacle course for scanner/viewer testing before modifying training.
- The semantic course must be static, terrain-attached, difficulty-aligned, and later migratable into the training trajectory config.
- The existing viewer and scanner path cannot currently show semantic obstacle hits or obstacle-surface elevation.

### Approved Decisions

- semantic stage is always enabled and tied to terrain difficulty
- four stages:
  - `S1`: none
  - `S2`: four small obstacles
  - `S3`: large plus small obstacles
  - `S4`: large plus more small obstacles
- sensor name must be `semantic_height_scanner`
- the viewer-derived config must delete inherited `height_scanner`
- semantic maps return `0=terrain`, `1=small`, `2=large`
- semantic obstacle generation belongs under `Go2Pvcnn/extension/semantic_course.py`
- semantic geometry must be created before sensor initialization, so `prestartup` replaces `startup`
- semantic viewer scene must set `replicate_physics = False`
- semantic-course root containers under `/World/semantic_course/{small,large}` must always exist
- semantic diagnostics count only valid sampled hits and must include an elevation-lift metric
- semantic rollout correctness is required on default `together`; `legacy` only needs a smoke if still exposed

### Sequencing Constraint

- `InteractiveScene` is built before manager loading.
- `prestartup` can mutate the USD stage before `sim.reset()`.
- sensors initialize their warp meshes on timeline `PLAY`, triggered by `sim.reset()`.
- `startup` happens after manager loading and is therefore too late for static semantic mesh inclusion.
- `prestartup` requires the semantic viewer scene to disable `replicate_physics`.

### Main-Agent Reserved Seam

- lifecycle and scene mode:
  - `replicate_physics = False`
  - `prestartup` spawning
- authoritative `ray_hits_w -> planner terrain` conversion
- deterministic viewer exposure to representative `S1..S4` rows

These seams stay under direct main-agent review even when implementation workers are dispatched.
