# T113a Semantic Support Consistency

## Meta

- Time: `2026-05-07 20:30 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T113a](../todo/T100-batched-together-planner-gpu-migration.md#t113a-semantic-valid-support-query-and-touchdownsupport-consistency)

## Purpose

- Verify the first execution leaf under `T113`: semantic-valid support filtering and touchdown/support `xy/z` consistency.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/terrain.py](../../Go2Pvcnn/extension/batched_together_planner/terrain.py)
- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_terrain.py](../../Go2Pvcnn/tests/test_batched_together_semantic_terrain.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## Subagent Result Summary

- `support_at(...)` now filters candidates to legal `terrain` cells only.
- Windows with no legal support return `NaN` sentinels instead of silently choosing obstacle surfaces.
- The touchdown support reconstruction path now uses support-resolved `xy` together with the corresponding support-derived `z`, closing the old mismatch path.
- Deterministic regressions were added/updated for:
  - legal-support filtering
  - no-legal-terrain failure signaling
  - support `xy/z` consistency
  - planner infeasible propagation when all support windows are obstacle-only

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_terrain.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py -k 'support_at_filters_obstacle_surfaces_and_returns_terrain_support or support_at_returns_nan_when_no_legal_terrain_exists or support_xy_z_consistency_regression_uses_same_support_solution or plan_segment_marks_all_obstacle_support_windows_infeasible'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_terrain.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
4. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/terrain.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_semantic_terrain.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Key Metrics

- targeted support/touchdown subset: `4 passed`
- full semantic terrain/planner focused suite: `14 passed`
- together guardrail subset: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `support_at(...)` never returns `small` / `large` as legal support
  - covered by `test_support_at_filters_obstacle_surfaces_and_returns_terrain_support`
- `support_xy_z_consistency` regression passes
  - covered by `test_support_xy_z_consistency_regression_uses_same_support_solution`
- no-legal-terrain windows fail/infeasible instead of obstacle fallback
  - query level: `test_support_at_returns_nan_when_no_legal_terrain_exists`
  - planner level: `test_plan_segment_marks_all_obstacle_support_windows_infeasible`

## Caveats

- Full repository `pytest` remains blocked by the existing `Go2Pvcnn/tests/conftest.py` dependency on missing `scripts.go2fp`; this leaf used the existing together-only `--noconftest` verification path.
- Infeasible propagation currently relies on the existing `compute_costs(...)` finite-mask behavior consuming `NaN` support sentinels rather than on a new explicit support-valid flag.

## Conclusion

- `T113a` is complete and verified within the current together-only test path.
- The next recommended execution step is `T113b`, so the always-on foothold-level `K=3` candidate axis builds on the new terrain-only support contract.

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 (2026-05-07 20:30 +0800); T113a code/test changes plus notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/terrain.py](../../Go2Pvcnn/extension/batched_together_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_terrain.py](../../Go2Pvcnn/tests/test_batched_together_semantic_terrain.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
