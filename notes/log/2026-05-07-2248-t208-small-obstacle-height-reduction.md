# T208 Small Obstacle Height Reduction

## Meta

- Time: `2026-05-07 22:48 +0800`
- Stage: `semantic static course geometry follow-up`
- Result: `pass with scoped runtime caveat`
- Todo: [T200/T208](../todo/T200-semantic-static-course-viewer.md#t208-small-obstacle-height-reduction-follow-up)

## Purpose

- Lower the semantic-course `small obstacle` height while preserving deterministic layout, grounding, and scanner/runtime contracts.
- Re-run focused tests using the new `small obstacle` geometry rather than only changing constants.

## Changes

- Lowered `SMALL_OBSTACLE_HEIGHT` in [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py) from `0.22` to `0.16`.
- Kept `SMALL_OBSTACLE_DIAMETER` unchanged at `0.12`.
- Updated [../../Go2Pvcnn/tests/test_semantic_course.py](../../Go2Pvcnn/tests/test_semantic_course.py) expectations for:
  - `semantic_scale_profile("small")`
  - native-shape parameter mapping for `small`
  - shape-aware grounding offsets implied by the lower `small` height
  - direct `build_course_anchors(...)` small-anchor target-height coverage

## Verification

- Red test before implementation:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_semantic_course.py -k 'semantic_scale_profile_matches_approved_sizes or shape_params_follow_native_mapping or grounding_offsets_are_shape_aware or small_anchor_targets_keep_diameter_but_use_lower_height_contract'`
  - result: `2 failed, 11 passed, 34 deselected`
  - expected failing evidence:
    - `semantic_scale_profile("small")` still returned `(0.12, 0.22)`
    - generated small anchors still had `target_height == 0.22`
- Focused semantic-course subset after implementation:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_semantic_course.py -k 'semantic_scale_profile_matches_approved_sizes or shape_params_follow_native_mapping or grounding_offsets_are_shape_aware or small_anchor_targets_keep_diameter_but_use_lower_height_contract'`
  - result: `13 passed`
- Full semantic-course suite:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_semantic_course.py -q`
  - result: `47 passed`
- Focused together semantic-planner small-obstacle path with the new geometry still present in repo:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 'small_low_obstacle_keeps_center_route or f2_small_forward_beyond_better_selects_beyond_small_terrain or f3_small_forward_front_better_may_legally_stay_before_obstacle'`
  - result: `3 passed, 11 deselected`
- Compile:
  - `python -m py_compile Go2Pvcnn/extension/semantic_course.py Go2Pvcnn/tests/test_semantic_course.py`
  - result: pass
- Focused real runtime small-anchor scan under the available Isaac env:
  - `conda run -n env_isaacsim python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_viewer_together_targeted_s4_small_scan_reports_semantic_hits -vv --tb=short`
  - result: process exit code `0`
  - observed output reached node start:
    - `Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_viewer_together_targeted_s4_small_scan_reports_semantic_hits`

## Metrics / Acceptance

- `small obstacle` height contract is now `0.16m`, with diameter still `0.12m`.
- Deterministic layout generation remains intact: full [../../Go2Pvcnn/tests/test_semantic_course.py](../../Go2Pvcnn/tests/test_semantic_course.py) passed.
- Shape-aware grounding remains valid under the reduced height: grounding offset tests passed.
- Focused planner-side small-obstacle semantics still pass after the geometry reduction.
- Focused runtime semantic scan still reaches the targeted `S4 small` anchor path under the currently available Isaac environment.

## Caveats

- This machine does not have the historical `env_isaaclab` conda environment referenced in older logs; the available runtime env is `env_isaacsim`.
- Repo-level runtime pytest without `--noconftest` is still blocked by the existing `Go2Pvcnn/tests/conftest.py` import of missing `scripts.go2fp`, so the focused runtime check used the same `--noconftest` workaround already used by together-focused verification.
- As with prior Isaac runtime logs in this repository, the targeted runtime test emitted collection and node-start output and exited `0` without a normal final `PASSED` summary line; this remains a scoped runtime-output caveat rather than a T208 blocker.

## Conclusion

- `T208` is complete at the intended semantic-course scope: `small obstacle` geometry is lower, deterministic semantic-course behavior is preserved, and focused semantic/planner/runtime checks were re-run against the new geometry.

## Git Refs

- Baseline Ref: `working tree on top of 130c635 with unrelated planner/viewer/note dirt present`
- Candidate Ref: `working tree with T208 semantic-course small-height change and focused verification`
- Key Files:
  - [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
  - [../../Go2Pvcnn/tests/test_semantic_course.py](../../Go2Pvcnn/tests/test_semantic_course.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
