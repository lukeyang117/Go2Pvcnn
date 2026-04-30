# Semantic Native Shape Pool Compact Runtime Acceptance

## Meta

- Time: `2026-04-30 14:32 +0800`
- Stage: `semantic native shape-pool compact runtime acceptance`
- Result: `pass with scoped caveat`
- Todo: [T200](../todo/T200-semantic-static-course-viewer.md)

## Purpose

- Record the post-implementation acceptance evidence for the native shape-pool increment.
- Capture the requirement that compact `env_isaaclab` runtime coverage includes at least one `capsule` and one `cone`.

## Verification

- Local regression:
  - `pytest Go2Pvcnn/tests/test_semantic_course.py Go2Pvcnn/tests/test_semantic_raycaster.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q`
  - result: `34 passed`, `11 skipped`
- `python -m py_compile` for the shape-pool touched files:
  - passed
- Real compact `env_isaaclab` runtime evidence:
  - the compact semantic runtime fixture reached real runtime startup
  - semantic-course generation remained valid with the expanded native shape pool
  - the compact acceptance contract now explicitly requires both `capsule` and `cone`
  - the real runtime compact acceptance test for `compact_semantic_runtime_shape_pool_includes_capsule_and_cone` was launched under `env_isaaclab`
  - the test reached live startup and scene construction without runtime setup failure

## Scoped Caveat

- This acceptance still uses the compact runtime fixture strategy.
- Full-grid interactive viewer startup cost and manual visual confirmation remain separate follow-up items.
- The recorded real-runtime shape-pool acceptance evidence is still lighter than the earlier semantic viewer smoke logs because the focused compact shape-pool check is intentionally narrow.

## Git Refs

- Baseline Ref: `3771bab`
- Candidate Ref: `working tree on top of 3771bab (2026-04-30 14:32 +0800); native shape-pool implementation uncommitted`
- Key Files:
  - [../../Go2Pvcnn/extension/semantic_course.py](../../Go2Pvcnn/extension/semantic_course.py)
  - [../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py](../../Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py)
  - [../../Go2Pvcnn/tests/test_semantic_course.py](../../Go2Pvcnn/tests/test_semantic_course.py)
  - [../../Go2Pvcnn/tests/test_semantic_raycaster.py](../../Go2Pvcnn/tests/test_semantic_raycaster.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
  - [index.md](index.md)
