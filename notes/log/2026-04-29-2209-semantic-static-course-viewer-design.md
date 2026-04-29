# Semantic Static Course Viewer Design

## Meta

- Time: `2026-04-29 22:09 +0800`
- Stage: `semantic static course viewer design`
- Result: `design recorded`
- Todo: [T200](../todo/T200-semantic-static-course-viewer.md)

## Purpose

- Record the user-approved design for a viewer-first semantic static obstacle course before implementation.
- Preserve the source-order finding that semantic props must exist before sensor initialization.
- Capture the approved semantic scanner contract and terrain-difficulty course rules for later migration into the training trajectory config.

## Design Decisions

- Keep training config unchanged in this phase; add a viewer-only derived config instead.
- Delete inherited `height_scanner` in the viewer config and replace it with `semantic_height_scanner`.
- `semantic_height_scanner` must return:
  - `elevation_map` at `1.5 x 1.5 m @ 0.01 m`
  - `semantic_map` with matching shape and semantic ids `0=terrain`, `1=small`, `2=large`
- Bind semantic course stages to terrain difficulty bands:
  - `S1`: none
  - `S2`: four small obstacles
  - `S3`: large plus small obstacles
  - `S4`: large plus more small obstacles
- Generate semantic props per terrain tile, not per env instance.
- Place semantic course logic in [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py).
- Treat `semantic_raycaster` redesign as part of scope:
  - support recursive geometry collection under semantic root paths
  - merge static terrain + semantic course geometry into one semantic mesh
  - keep tests for shape, ids, and obstacle-surface elevation
- Update viewer playback to color terrain, small-obstacle, and large-obstacle hits differently.

## Verification

- Source-inspection verification only; no runtime code changed yet.
- Confirmed from Isaac Lab source that sensor initialization occurs on `sim.reset()` / timeline `PLAY` before `startup`, so `startup` cannot be used for semantic prop creation.
- Confirmed from `InteractiveScene` source that inherited `height_scanner` can be removed by overriding it with `None` in the derived scene config.
- Spec, todo dashboard, T200 branch page, and log index are updated.

## Git Refs

- Baseline Ref: `6279bc4`
- Candidate Ref: `working tree on top of 6279bc4 (2026-04-29 22:09 +0800); spec + notes design update`
- Key Files:
  - [../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md](../../docs/superpowers/specs/2026-04-29-semantic-static-course-viewer-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [index.md](index.md)
