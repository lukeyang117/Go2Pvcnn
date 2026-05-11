# Semantic-Aware Together Viewer / Planner Implementation

## Metadata

- Time: 2026-05-06 23:40 +0800
- Topic: T112 semantic-aware together planner + viewer row/col targeting
- Stage: extension/batched_together_planner + extension/viz + focused verification
- Related Todo:
  - [T100/T112](../todo/T100-batched-together-planner-gpu-migration.md#t112-semantic-aware-together-planner--viewer-rowcol-targeting)
- Baseline Ref: `cf7e9cf`
- Candidate Ref: working tree on top of `cf7e9cf`

## Purpose

Implement the first semantic-aware `together` planner path for the viewer:

- shared semantic terrain/query ABI
- semantic candidate selection and semantic cost terms
- viewer `--terrain-row` / `--terrain-col` targeting
- manager semantic-map forwarding
- focused tests and GPU guardrail confirmation

## Key Files

- [../../Go2Pvcnn/extension/batched_together_planner/terrain.py](../../Go2Pvcnn/extension/batched_together_planner/terrain.py)
- [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
- [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
- [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
- [../../Go2Pvcnn/extension/batched_together_planner/manager.py](../../Go2Pvcnn/extension/batched_together_planner/manager.py)
- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_terrain.py](../../Go2Pvcnn/tests/test_batched_together_semantic_terrain.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
- [../../Go2Pvcnn/tests/test_batched_together_parity.py](../../Go2Pvcnn/tests/test_batched_together_parity.py)
- [../../Go2Pvcnn/tests/test_batched_together_runtime_path.py](../../Go2Pvcnn/tests/test_batched_together_runtime_path.py)
- [../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py](../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py)
- [../../Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
- [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)

## What Changed

1. Shared semantic terrain ABI
   - `TogetherPlannerTerrain` now supports optional `semantic_maps`
   - added `semantic_at`, `obstacle_mask_at`, `obstacle_height_at`, `terrain_reference_height_at`, `obstacle_relative_height_at`
   - added shared extractor `build_together_terrain_from_scanner(...)`

2. Semantic planner behavior
   - added semantic config knobs to `TogetherPlannerConfig`
   - added fixed 3-route semantic candidate expansion in `plan_segment(...)`
   - added semantic touchdown/swing/body/route costs
   - preserved height-only path with zero/default semantic diagnostics
   - added stable result diagnostics:
     - `selected_route_offset`
     - `semantic_candidate_costs`

3. Manager compatibility
   - `TogetherTrajectoryManager._terrain_from_env(...)` now forwards semantic maps through the shared extractor

4. Viewer targeting
   - `go2_foostep_planner.py` now accepts:
     - `--terrain-row`
     - `--terrain-col`
   - env0 reset/spawn can be retargeted to a selected `terrain_origins[row, col]`
   - scanner refresh helper keeps targeted reset scans synchronized

## Verification

### Passed

1. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/manager.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/terrain.py Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_parity.py Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_batched_together_semantic_terrain.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_viz_playback.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py`
2. `python -m unittest Go2Pvcnn.tests.test_batched_together_guardrails Go2Pvcnn.tests.test_batched_planner_runtime_path.BatchedPlannerRuntimePathTest.test_viewer_parser_defaults_match_validated_diagnostics_regime Go2Pvcnn.tests.test_batched_planner_runtime_path.BatchedPlannerRuntimePathTest.test_viewer_parser_accepts_explicit_terrain_row_col`
3. `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_terrain.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_batched_together_core.py -q`
   - result: `16 passed`
4. `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viz_playback.py -q`
   - result: `33 passed`
5. `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_runtime_path.py -q`
   - result: `11 passed`
6. Focused direct behavior checks with `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn`
   - viewer helper checks: passed
   - semantic large-obstacle route-selection check: passed

### Failed / Blocked

1. Repository-style `pytest ...` runs that load [../../Go2Pvcnn/tests/conftest.py](../../Go2Pvcnn/tests/conftest.py)
   - blocker: missing `raw/kinematic_footsteps` checkout provides no `scripts.go2fp.config`
   - observed error: `ModuleNotFoundError: No module named 'scripts.go2fp'`
   - this is an environment/repository dependency issue, not a direct failure in the modified semantic-aware viewer/together slice

## Result

- Focused semantic/viewer/together verification is green
- together guardrail is green after removing one loop from planner-side breakdown gathering
- full repository-wide pytest remains partially blocked by missing raw planner dependencies

## Follow-Up

- run the semantic viewer interactively on a chosen tile for visual confirmation once the user wants runtime inspection
- if full-suite pytest is required, restore the missing `raw/kinematic_footsteps` dependency path first
