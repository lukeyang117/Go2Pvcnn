# T115a Grounded Touchdown State-Contract Tightening

## Meta

- Time: `2026-05-08 21:03 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T115a](../todo/T100-batched-together-planner-gpu-migration.md#t115a-grounded-touchdown-metrics-and-state-contract-tightening)

## Purpose

- Implement the deterministic/core portion of `T115a` only.
- Tighten `front_cross` / `rear_follow` / `clear` semantics around grounded touchdown contracts.
- Expose grounded touchdown diagnostics on rollout and selected result surfaces without touching the runtime harness leaves.

## Scope

- Code changed only in:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
- No runtime harness or viewer diagnostics files were modified in this leaf.

## Implementation Notes

- Added `touchdown_ground_gap_tolerance_m = 0.02` to the together config.
- Added rollout/result diagnostics:
  - `front_touchdown_ground_gap`
  - `rear_touchdown_ground_gap`
- Ground-gap reference uses the selected support solution height from semantic touchdown support, not plain terrain height.
- Tightened corridor state semantics:
  - `front_cross` now additionally requires grounded front touchdown.
  - `rear_follow` now additionally requires grounded front and rear touchdown.
  - `clear` is now reachable only after both front and rear grounded conditions hold.
- Tightened crossing acceptance:
  - crossing-state candidates are infeasible if grounded front+rear completion is not satisfied
  - selected `small_strategy_outcome` degrades away from crossing when airborne touchdown breaks the grounded contract

## Verification

### TDD red

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't115a_'`
- Initial result before implementation:
  - `5 failed`
- Initial failure surface:
  - missing `front_touchdown_ground_gap`
  - missing `rear_touchdown_ground_gap`

### Focused T115a

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't115a_'`
- Result:
  - `6 passed`

### Affected semantic union

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
- Result:
  - `45 passed`

### Affected core union

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
- Result:
  - `6 passed`

### Guardrail rerun

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
- Result:
  - `5 passed`

## Acceptance Coverage

- `front_touchdown_ground_gap` exposed and asserted:
  - covered by focused rollout/result tests
- `rear_touchdown_ground_gap` exposed and asserted:
  - covered by focused rollout/result tests
- `clear` unreachable before grounded front/rear completion:
  - covered by `test_t115a_clear_requires_grounded_front_and_rear_completion`
- airborne rear touchdown causes crossing acceptance to fail:
  - covered by `test_t115a_selected_crossing_outcome_fails_when_touchdown_is_airborne`
  - selected outcome degrades away from crossing when the rear touchdown gap exceeds `0.02 m`

## Conclusion

- `T115a` deterministic/core scope is implemented and verified.
- `T115b+` remain open for three-surface crossing validity and runtime/headless Isaac Lab acceptance.

## Follow-up

- Continue with `T115b` next; keep scope limited to touchdown/foot/base three-surface crossing validity.
- Do not treat this leaf as runtime acceptance for `env_isaacsim`; that remains with `T115d/T115e`.

## Git Refs

- Baseline Ref: `working tree before T115a deterministic/core grounded-contract patch`
- Candidate Ref: `working tree with T115a grounded gap diagnostics + grounded state-contract tightening`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
