# Semantic Viewer Empty Marker Fix

## Meta

- Time: `2026-04-30 02:15 +0800`
- Stage: `semantic viewer empty marker fix`
- Result: `pass`
- Todo: [T200](../todo/T200-semantic-static-course-viewer.md)

## Purpose

- Fix the interactive semantic viewer crash triggered when one semantic class had zero sampled points and `VisualizationMarkers.visualize()` was called with an empty translation tensor.
- Apply the user-requested semantic colors:
  - terrain: white
  - small obstacle: green
  - large obstacle: red

## Root Cause

- `PlannerVisualizer.update()` iterated over every semantic marker group and called `markers.visualize(translations=empty)` when a class had zero points.
- Isaac Lab `VisualizationMarkers.visualize()` rejects zero markers and raises:
  - `ValueError: Number of markers cannot be zero!`

## Fix

- When a semantic class has zero points:
  - call `markers.set_visibility(False)`
  - do **not** call `visualize(...)`
- When a class has points:
  - call `markers.set_visibility(True)`
  - then call `visualize(...)`
- Updated semantic colors:
  - terrain `(1.0, 1.0, 1.0)`
  - small `(0.2, 0.9, 0.2)`
  - large `(1.0, 0.2, 0.2)`

## Verification

- `pytest Go2Pvcnn/tests/test_viz_playback.py -q -k "planner_visualizer_hides_empty_semantic_classes_without_zero_marker_visualize or subsample_semantic_height_points or format_semantic_diagnostics"`
  - `3 passed`
- `pytest Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q`
  - `13 passed`, `10 skipped`

## Git Refs

- Baseline Ref: `a9faabe`
- Candidate Ref: `working tree on top of a9faabe (2026-04-30 02:15 +0800); semantic viewer empty-marker crash fix pending commit`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
  - [index.md](index.md)
