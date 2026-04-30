# Semantic Course Random Layout Embedded Plan

## Meta

- Time: `2026-04-30 15:22 +0800`
- Stage: `semantic static course implementation planning`
- Result: `plan recorded in branch memory`
- Todo: [T200/T207](../todo/T200-semantic-static-course-viewer.md#t207-deterministic-full-sub-terrain-semantic-layout--footprint-grounding)

## Purpose

- Record the user's requested implementation-planning shape without creating a separate `docs/superpowers/plans/` document.
- Capture master/sub-agent ownership, parallelization constraints, and test metrics for T207.

## Plan Summary

- Master-agent controls orchestration, reviews, integration, final verification, and notes/log updates.
- Sub-agent W1 owns layout API, deterministic random anchors, tile-size resolution, and layout tests.
- Sub-agent W2 owns footprint grounding and grounding tests after W1 lands.
- Sub-agent W3 owns targeted runtime semantic-hit support after W1's API is stable.
- Reviews are two-stage:
  - spec compliance against [../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md](../../docs/superpowers/specs/2026-04-30-semantic-course-random-layout-grounding-design.md)
  - code quality and integration risk review
- W1/W2 should not write concurrently because both touch [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py).
- Read-only exploration or review/test-only agents may run in parallel when their scopes do not write the same files.

## Verification Targets

- `pytest Go2Pvcnn/tests/test_semantic_course.py -q`
- `pytest Go2Pvcnn/tests/test_semantic_raycaster.py -q`
- `pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q` in `/home/lhy/anaconda3/envs/env_isaaclab`, or documented runtime-resource caveat
- Pure layout assertions:
  - default S4 spreads outside the old center scanner footprint
  - same seed reproduces layout
  - different tiles differ
  - canonical `8m x 8m` case does not use fallback
- Pure grounding assertions:
  - footprint max height minus `0.015m` plus shape offset controls default z
- Runtime semantic assertions:
  - targeted S4 small reports semantic id `1`
  - targeted S4 large reports semantic id `2`
  - shape-pool coverage still includes `capsule` and `cone`

## Git Refs

- Baseline Ref: `6b6d80d`
- Candidate Ref: `working tree on top of 6b6d80d (2026-04-30 15:22 +0800); embedded plan update uncommitted`
- Key Files:
  - [../todo/T200-semantic-static-course-viewer.md](../todo/T200-semantic-static-course-viewer.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
