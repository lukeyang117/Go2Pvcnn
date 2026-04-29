# T200 Semantic Static Course Viewer

## Current State

- User-approved design direction is recorded for a viewer-first semantic static obstacle course tied to terrain difficulty.
- Training config remains unchanged for now; the immediate target is a derived viewer config that can later be migrated into `teacher_elevation_trajectory_env_cfg.py`.
- The old inherited `height_scanner` must be removed in the viewer config and replaced by `semantic_height_scanner`.
- `semantic_raycaster` is in scope for redesign and tests, not just reuse.
- Source inspection established that `startup` is too late for semantic prop spawning because sensor warp-mesh initialization happens on `sim.reset()` / timeline `PLAY` before `startup`. Semantic props must exist by `prestartup`.

## Open Children

- T201: redesign `semantic_raycaster` root traversal / static semantic merge / sensor tests
- T202: add `extension/semantic_course.py` for difficulty-to-stage mapping, static tile layouts, and grounded cuboid generation
- T203: add `teacher_elevation_trajectory_semantic_viewer_env_cfg.py` and replace inherited scanner references with `semantic_height_scanner`
- T204: update `go2_foostep_planner.py` to consume `semantic_height_scanner` and color semantic hits

## Closed Children Archive

- none yet

## Related Logs

- [2026-04-29-2209-semantic-static-course-viewer-design.md](../log/2026-04-29-2209-semantic-static-course-viewer-design.md)

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

- Run the spec review loop on the new design document.
- If the spec review passes, ask the user to review the written spec before creating the implementation plan.

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

### Sequencing Constraint

- `InteractiveScene` is built before manager loading.
- `prestartup` can mutate the USD stage before `sim.reset()`.
- sensors initialize their warp meshes on timeline `PLAY`, triggered by `sim.reset()`.
- `startup` happens after manager loading and is therefore too late for static semantic mesh inclusion.
