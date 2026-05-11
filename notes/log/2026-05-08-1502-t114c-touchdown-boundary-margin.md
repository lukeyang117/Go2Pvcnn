# T114c Touchdown Boundary Margin

## Meta

- Time: `2026-05-08 15:02 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T114c](../todo/T100-batched-together-planner-gpu-migration.md#t114c-touchdown-boundary-margin-controls)

## Purpose

- Verify the third execution leaf under `T114`: explicit touchdown-to-small-boundary margin control with penalty/invalidation behavior, while preserving endpoint legality and pure-GPU fixed-shape constraints.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
- [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114c_'`
  - result: `2 failed, 23 deselected`
  - failing reason: `TogetherRollout` did not yet expose `touchdown_small_margin`
- GREEN after minimal implementation:
  - the same `t114c_` subset rerun passed

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114c_'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
5. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Key Metrics

- TDD red subset: `2 failed`
- focused `T114c` subset after implementation: `2 passed`
- semantic affected-union rerun: `25 passed`
- core affected-union rerun: `6 passed`
- guardrail rerun: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `touchdown_small_margin` is now computed and asserted
  - covered by `test_t114c_touchdown_small_margin_is_computed_on_k3_rollout`
- `F15_near_boundary_penalize_or_invalidate` passes
  - covered by `test_t114c_f15_near_boundary_penalize_or_invalidate`
- touchdown still never lands on small/large surfaces
  - preserved by explicit surface-count assertions and final semantic affected-union rerun
- guardrail suite remained green, so the new margin logic did not violate pure-GPU/no-`for`/no-CPU hot-path constraints

## Caveats

- `touchdown_small_margin` currently lives on `TogetherRollout`, not yet on `TogetherPlannerResult`; that matches this leaf’s scope, but later leaves may decide to surface it farther outward if runtime-facing diagnostics become necessary.
- The margin thresholds are intentionally heuristic tuning knobs at this stage and may still be refined by later leaves when pair-consistency and path-clearance logic are added.

## Conclusion

- `T114c` is complete and verified with TDD evidence plus final affected-union reruns on the final code state.
- Recommended next leaf is `T114d`.

## Git Refs

- Baseline Ref: `current working tree after T114b verification`
- Candidate Ref: `working tree with T114c boundary-margin changes and focused verification; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
