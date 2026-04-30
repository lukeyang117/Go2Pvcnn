# Semantic Course Layout/Grounding Implementation

## Metadata

- Time: 2026-04-30 15:48 +0800
- Todo: [T200/T207](../todo/T200-semantic-static-course-viewer.md#t207-deterministic-full-sub-terrain-semantic-layout--footprint-grounding)
- Stage: semantic static-course implementation
- Result: layout and footprint-grounding tasks landed; runtime targeted scan task still in progress
- Commits:
  - `1628df0 feat: randomize semantic course tile layouts`
  - `0a81e4d feat: ground semantic course by footprints`

## What Changed

- `extension/semantic_course.py` now builds deterministic per-tile semantic layouts instead of fixed center-cluster coordinates.
- S1-S4 semantic counts are preserved, while S2-S4 anchors sample across most of each tile with margin, center safety, spacing, and bounded fallback diagnostics.
- Tile size resolution now prefers terrain-generator size, then origin spacing, then the `8m x 8m` fallback.
- Grounding now samples each obstacle footprint at center plus eight support points and defaults to `max(finite footprint height) - 0.015m + shape bottom-to-center offset`.
- Runtime grounding batches all footprint points through the existing terrain raycast sampler and honors `SemanticCourseGroundingCfg`.

## Review Gates

- W1 layout spec review: approved.
- W1 layout code-quality review: approved.
- W2 grounding spec review: approved.
- W2 grounding code-quality review: approved.
- Reviewer non-blocking suggestions addressed before W2 commit:
  - `embed_depth_m` finite/non-negative validation
  - monkeypatched runtime sampler test for footprint batching and non-default grounding config

## Verification

- RED W1: `pytest Go2Pvcnn/tests/test_semantic_course.py -q`
  - failed during collection before `DEFAULT_CENTER_SAFETY_HALF_EXTENT_M` existed
- GREEN W1:
  - `pytest Go2Pvcnn/tests/test_semantic_course.py -q` -> `32 passed`
  - `python -m py_compile Go2Pvcnn/extension/semantic_course.py Go2Pvcnn/tests/test_semantic_course.py` -> pass
- RED W2:
  - `pytest Go2Pvcnn/tests/test_semantic_course.py -q`
  - failed during collection before `footprint_sample_offsets` existed
- RED reviewer follow-up:
  - `pytest Go2Pvcnn/tests/test_semantic_course.py -q`
  - `test_ground_course_anchors_rejects_invalid_embed_depth` failed because negative `embed_depth_m` was accepted
- GREEN W2:
  - `pytest Go2Pvcnn/tests/test_semantic_course.py -q` -> `46 passed in 0.15s`
  - `python -m py_compile Go2Pvcnn/extension/semantic_course.py Go2Pvcnn/tests/test_semantic_course.py` -> pass

## Caveats / Next

- Targeted runtime scan support is still in progress under T207 task 3.
- Existing unrelated dirty workspace entries were left untouched:
  - `raw/kinematic_footsteps`
  - `${data}/NvStreamer-*.etli`
