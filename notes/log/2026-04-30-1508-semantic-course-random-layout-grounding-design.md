# Semantic Course Random Layout Grounding Design

## Meta

- Time: `2026-04-30 15:08 +0800`
- Stage: `semantic static course design`
- Result: `design recorded`
- Todo: [T200/T207](../todo/T200-semantic-static-course-viewer.md#t207-deterministic-full-sub-terrain-semantic-layout--footprint-grounding)

## Purpose

- Record the user-approved design direction for spreading semantic objects across each sub-terrain instead of concentrating them near the center scanner window.
- Capture the selected grounding policy for irregular terrain.

## Design Decisions

- Primary implementation target remains [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py).
- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py) should keep consuming `semantic_height_scanner`; it should not own layout logic.
- Use full sub-terrain distribution rather than guaranteeing default spawn-window semantic visibility.
- Use deterministic pseudo-random layout keyed by seed, stage, row, col, slot, and semantic class.
- Keep current S1-S4 object counts and semantic classes.
- Sample tile-local xy across most of the `8m x 8m` tile with border margin, center safety, and minimum spacing constraints.
- Ground objects upright with footprint multi-point terrain sampling.
- Use a high robust terrain height minus a small embedding depth, then add the existing shape-aware bottom-to-center offset.
- Automated semantic-hit runtime checks should target a known generated obstacle if the default spawn scanner no longer sees semantic objects.

## Verification

- Design-only step; no implementation code changed yet.
- Spec written at [../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md](../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md).
- Notes, branch memory, and log index are updated.

## Git Refs

- Baseline Ref: `ec7efe2`
- Candidate Ref: `working tree on top of ec7efe2 (2026-04-30 15:08 +0800); design notes update`
- Key Files:
  - [../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md](../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md)
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
