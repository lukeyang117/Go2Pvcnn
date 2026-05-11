# T114f State-Aware Selection

## Meta

- Time: `2026-05-08 16:34 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T114f](../todo/T100-batched-together-planner-gpu-migration.md#t114f-state-aware-invalidation-and-selection-rules)

## Purpose

- Verify the sixth execution leaf under `T114`: state-aware invalidation and selection rules for `small` approach/cross/bypass and `large` bypass/refusal.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
- [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
- [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
- [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114f_'`
  - result: `4 failed, 31 deselected`
  - failing reason: `TogetherPlannerResult` did not yet expose `state_mode` / `small_strategy_outcome`
- GREEN after minimal implementation:
  - the same `t114f_` subset rerun passed

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114f_'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
5. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Key Metrics

- TDD red subset: `4 failed`
- focused `T114f` subset after implementation: `4 passed`
- semantic affected-union rerun: `35 passed`
- core affected-union rerun: `6 passed`
- guardrail rerun: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `state_mode` and `small_strategy_outcome` are now explainable on selected results
- deterministic crossing-vs-approach selection is covered
- deterministic small-bypass selection is covered
- large still prefers bypass/refusal
- `front_cross`, `rear_follow`, and `clear` remain explicitly covered through the state fixtures already in the affected union
- state-aware candidate selection now prefers:
  - feasible candidates over infeasible ones
  - safe-fallback candidates over fully invalid ones
  - non-center bypass candidates over center ones in bypass state

## Caveats

- This leaf required additional directly-related files beyond the minimal scope:
  - `types.py`
  - `config.py`
  - `costs.py`
- `small_strategy_outcome` is intentionally a compact explainability signal, not yet a full runtime-facing state-machine trace.
- The new `state_bypass_center_penalty_weight` remains a heuristic tuning knob and may need retuning after final traceability closure.

## Conclusion

- `T114f` is complete and verified with TDD evidence plus final affected-union reruns on the final code state.
- Recommended next leaf is `T114g`.

## Git Refs

- Baseline Ref: `current working tree after T114e verification`
- Candidate Ref: `working tree with T114f state-aware selection changes and focused verification; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
