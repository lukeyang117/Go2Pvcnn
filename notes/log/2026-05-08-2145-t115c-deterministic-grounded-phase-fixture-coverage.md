# T115c Deterministic Grounded-Phase Fixture Coverage

## Meta

- Time: `2026-05-08 21:45 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T115c](../todo/T100-batched-together-planner-gpu-migration.md#t115c-deterministic-grounded-phase-fixture-coverage)

## Purpose

- Close the deterministic-only `T115c` leaf by proving grounded phase coverage for `G1-G5`.
- Distinguish whether `G4` was a fixture-design problem or a real candidate-selection semantic gap.
- Keep the work inside planner/test scope only; do not enter the `env_isaacsim` runtime leaves.

## Changes

- Added explicit `T115` deterministic fixture and metric traceability maps in [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py).
- Added deterministic coverage tests for:
  - `G1_front_cross_grounded`
  - `G2_rear_follow_grounded`
  - `G4_cross_degrades_to_bypass_when_rear_follow_not_groundable`
  - `G5_front_cross_then_rear_follow_then_clear`
- Kept `G3_rear_follow_airborne_invalid` mapped to the existing selected-result airborne invalidation test from `T115a`.
- Added a planner-side selection rerank in [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py) so that when the selected outcome is `BYPASS`, feasible lateral non-crossing bypass candidates are preferred over center-path obstacle-crossing candidates.

## Candidate-Level Diagnosis

- Initial red test showed the default `G4` scene did **not** prove a real bypass fallback:
  - selected outcome became `BYPASS`
  - but `selected_route_offset == 0`
  - and `base_path_crosses_small_flag == 1`
- Candidate inspection then separated two cases:
  - the default `G4` fixture often had no feasible lateral bypass candidate, so it was not a good deterministic proof of the intended contract
  - a nearby deterministic configuration did expose a feasible lateral bypass candidate while `plan_segment()` still selected center
- Final `G4` fixture was updated to that deterministic configuration:
  - `semantic_lateral_offset_m = 0.08`
  - obstacle height `0.08`
  - obstacle center `x = 0.18`
  - root `z = 0.28`
- That fixture now proves the selected result is a **real** lateral bypass:
  - `small_strategy_outcome == BYPASS`
  - `abs(selected_route_offset) > 0`
  - `base_path_crosses_small_flag == 0`
  - rear touchdown remains airborne enough to fail grounded crossing acceptance

## Verification

### TDD Red

- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't115a_ or t115b_ or t115c_'`
- First red state:
  - `1 failed, 14 passed, 39 deselected`
  - failing test: `test_t115c_g4_cross_degrades_to_bypass_when_rear_follow_not_groundable`

### Focused Green

- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't115a_ or t115b_ or t115c_'`
- Result:
  - `15 passed, 39 deselected`

### Final-Code-State Minimal Union

- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
  - `54 passed`
- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
  - `6 passed`
- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
  - `5 passed`

## Acceptance Coverage

- `G1_front_cross_grounded`
  - covered by `test_t115c_g1_front_cross_grounded_fixture_maps_to_front_cross_state`
  - plus grounded front/rear gap exposure from `test_t115a_front_cross_and_rear_gap_metrics_are_exposed`
- `G2_rear_follow_grounded`
  - covered by `test_t115c_g2_rear_follow_grounded_fixture_maps_to_rear_follow_state`
  - plus selected-result grounded gap exposure from `test_t115a_selected_result_exposes_grounded_touchdown_metrics`
- `G3_rear_follow_airborne_invalid`
  - covered by `test_t115a_selected_crossing_outcome_fails_when_touchdown_is_airborne`
- `G4_cross_degrades_to_bypass_when_rear_follow_not_groundable`
  - covered by `test_t115c_g4_cross_degrades_to_bypass_when_rear_follow_not_groundable`
  - selected result now proves true lateral bypass rather than center-path relabeling
- `G5_front_cross_then_rear_follow_then_clear`
  - covered by `test_t115c_g5_front_cross_then_rear_follow_then_clear_progression_is_stable`
  - phase sequence asserted explicitly as `front_cross -> rear_follow -> clear`

## Conclusion

- `T115c` is complete on the deterministic planner/test layer.
- The leaf required both:
  - a better deterministic `G4` fixture
  - a minimal planner selection fix so real bypass candidates win over center-path pseudo-bypass outcomes
- Runtime harness and Isaac Lab acceptance remain intentionally untouched for `T115d/T115e`.

## Follow-up

- Continue with `T115d` for headless `env_isaacsim` diagnostics surfacing.
- Keep the new deterministic `G4` fixture as the handoff point for runtime bypass authority.

## Git Refs

- Baseline Ref: `working tree with T115a/T115b complete; T115c not yet closed`
- Candidate Ref: `working tree with T115c deterministic fixture coverage, planner selection rerank, and final focused+union verification`
- Key Files:
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
