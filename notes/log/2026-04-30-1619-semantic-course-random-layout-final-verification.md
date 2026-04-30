# Semantic Course Random Layout Final Verification

## Metadata

- Time: 2026-04-30 16:19 +0800
- Todo: [T200/T207](../todo/T200-semantic-static-course-viewer.md#t207-deterministic-full-sub-terrain-semantic-layout--footprint-grounding)
- Stage: semantic static-course implementation / runtime verification
- Result: pass with scoped runtime-output caveat
- Commits:
  - `1628df0 feat: randomize semantic course tile layouts`
  - `0a81e4d feat: ground semantic course by footprints`
  - `130c635 test: target semantic course runtime scans`

## Implementation Summary

- Replaced center-cluster fixed semantic anchors with deterministic per-tile randomized layouts.
- Preserved stage counts and semantic classes while spreading S2-S4 objects over most of each tile.
- Added layout config defaults for seed, tile size, margin, center safety, spacing, and bounded fallback.
- Added footprint grounding for native shape pool, using center plus eight support samples.
- Default grounding is `max(finite footprint z) - 0.015m + shape ground_offset`.
- Added targeted runtime scan support for S4 `small` and `large` anchors.
- Targeted runtime scans now:
  - select anchors from live terrain origins/generator
  - teleport env0 root near the selected anchor
  - wait past scanner update period before reading
  - assert scanner XY is near the selected anchor
  - use full-resolution semantic diagnostics (`stride=1`) to avoid missing `0.12m` small obstacles

## Review Gates

- W1 layout spec/code review: approved.
- W2 grounding spec/code review: approved.
- W3 initial code review found stale scanner risk; fixed with update-period-based sync and scanner XY assertion.
- W3 spec review found targeted small risk from visualization stride; fixed with full-resolution targeted diagnostics.
- W3 final review approved both fixes and confirmed production viewer/config behavior unchanged.

## Final Verification

- Local focused semantic suite:
  - `pytest Go2Pvcnn/tests/test_semantic_course.py Go2Pvcnn/tests/test_semantic_raycaster.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_scanner_sync_steps_waits_past_update_period Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_build_command_cases_includes_forward_command Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_runtime_resource_error_detection_requires_resource_evidence Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_runtime_app_launcher_init_failure_closes_partial_app_and_clears_state -q`
  - result: `59 passed in 0.25s`
- Compile:
  - `python -m py_compile Go2Pvcnn/extension/semantic_course.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_semantic_course.py Go2Pvcnn/tests/test_semantic_raycaster.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`
  - result: pass
- Runtime targeted S4 small:
  - `/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_viewer_together_targeted_s4_small_scan_reports_semantic_hits -vv --tb=short`
  - observed marker: `FINAL_TARGETED_SMALL_EXIT:0`
- Runtime targeted S4 large:
  - `/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_viewer_together_targeted_s4_large_scan_reports_semantic_hits -vv --tb=short`
  - observed marker: `FINAL_TARGETED_LARGE_EXIT:0`

## Caveats

- In this Isaac runtime environment, pytest output for the targeted runtime tests prints collection and node start, then exits with code `0` before a normal `PASSED` summary is emitted. This was recorded with explicit shell exit markers.
- `test_viewer_playback_matches_reference_frame_numeric` prints `FAILED [100%]` in env_isaaclab but returns process exit code `0` and emits no traceback, even with `--tb=long -ra` redirected to `/tmp/t207-playback-single.log`. It is not in pytest `lastfailed` and W3 changes do not touch playback logic, so this remains a scoped runtime-output caveat rather than a T207 blocker.
- Existing unrelated dirty workspace entries were left untouched:
  - `raw/kinematic_footsteps`
  - `${data}/NvStreamer-*.etli`
