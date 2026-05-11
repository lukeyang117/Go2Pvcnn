# T115f Carry-Forward Union And Final-Code-State Authority

## Meta

- Time: `2026-05-08 22:39 +0800`
- Stage: `final-code-state authoritative rerun`
- Result: `pass`
- Todo: [T100/T115f](../todo/T100-batched-together-planner-gpu-migration.md#t115f-carry-forward-union-and-final-code-state-authority)

## Purpose

- Close `T115f` only.
- Convert `T113/T114/T115` carry-forward obligations into an explicit final authority record at the test layer.
- Replace earlier overlapping focused passes as final evidence with one fresh final-code-state rerun surface.

## Scope

- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
- [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
- [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
- [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
- [index.md](index.md)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 'test_t115f_final_authority_record_carry_forward_union_is_explicit'`
  - result: `1 failed, 54 deselected`
  - failing reason: missing `T115_FINAL_AUTHORITY_RECORD`
- GREEN after minimal implementation:
  - targeted authority/traceability subset rerun passed:
    - `5 passed, 54 deselected`

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
4. `timeout -s INT -k 20s 300s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r1_small_cross_runtime_grounded or test_r2_small_bypass_runtime or test_r3_rear_touchdown_airborne_regression or test_r4_runtime_clear_requires_grounded_completion or test_grounded_crossing_runtime_sequence_report_summarizes_acceptance_fields"'`
5. `python -m py_compile Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py`

## Key Metrics

- targeted `T115f` authority subset after implementation: `5 passed`
- semantic deterministic authoritative union: `59 passed`
- core deterministic authoritative union: `6 passed`
- guardrail authoritative union: `5 passed`
- runtime minimal authoritative union: `EXIT_CODE:0`
- runtime minimal authoritative union log: `/tmp/t115f-runtime-union-93ht.log`
- `py_compile`: `pass`

## Acceptance Coverage

- carry-forward `T113` mandatory gates are now explicit test metadata:
  - fixtures `F1-F9`
  - metrics `candidate_count`, `touchdown_semantic_valid_ratio`, `small_surface_touchdown_count`, `large_surface_touchdown_count`, `small_cross_preference_outcome`, `large_forward_refusal_ratio`, `body_min_clearance`, `leg_min_clearance`, `collision_penalty_breakdown`, `support_xy_z_consistency`, `forward_progress_metric`
- carry-forward `T114` mandatory gates remain explicit and are pulled into the final authority record:
  - fixtures `F1`, `F1b`, `F2-F15`
  - metrics including `state_mode`, `touchdown_small_margin`, `front_pair_consistency`, `rear_pair_follow_consistency`, `body_posture_score`, `anchor_to_touchdown_foot_clearance`, `anchor_to_touchdown_leg_clearance`, `candidate_path_collision_flag`, `small_strategy_outcome`, `candidate_action_segment_diagnostics_present`
- new `T115` deterministic gates are explicit:
  - fixtures `G1-G5`
  - grounded / three-surface metrics including `front_touchdown_ground_gap`, `rear_touchdown_ground_gap`, `front_cross_grounded_ratio`, `rear_follow_grounded_ratio`, `rear_touchdown_airborne_count`, `cross_phase_progression_valid`, `cross_outcome_grounded`, `touchdown_on_small_count`, `front_foot_small_collision_count`, `rear_foot_small_collision_count`, `front_foot_min_clearance_to_small`, `rear_foot_min_clearance_to_small`, `base_small_penetration_count`, `base_min_clearance_to_small`, `base_path_crosses_small_flag`
- final runtime/headless authority is explicit and executable:
  - `R1_small_cross_runtime_grounded`
  - `R2_small_bypass_runtime`
  - `R3_rear_touchdown_airborne_regression`
  - `R4_runtime_clear_requires_grounded_completion`
  - `test_grounded_crossing_runtime_sequence_report_summarizes_acceptance_fields`
- final authority policy is explicit:
  - `final code state only`
  - older overlapping focused passes are `superseded / non-authoritative`

## Runtime Result

- timeout-wrapped runtime minimal union returned:
  - `EXIT_CODE:0`
- captured runtime pytest output in `/tmp/t115f-runtime-union-93ht.log` was:
  - `.`
- this final authority run therefore does not depend on earlier `T115e` focused passes, even though those earlier logs remain useful background evidence

## Caveats

- Full repository-wide `pytest` without `--noconftest` remains blocked by the historical `Go2Pvcnn/tests/conftest.py` dependency on missing `scripts.go2fp`.
- The runtime authoritative command remains timeout-wrapped on this machine because that is still the most reliable way to force explicit terminal status under headless Isaac Lab.
- This leaf intentionally did not modify planner hot-path semantics; closure is at the test/traceability/authority layer only.

## Conclusion

- `T115f` is complete and verified.
- `T115` now has authoritative final-code-state evidence across deterministic/core unions, guardrails, and runtime/headless acceptance.
- `T115` can be treated as closed or verify-closed by the main agent based on this final authority log.

## Git Refs

- Baseline Ref: `working tree after T115e runtime acceptance closure`
- Candidate Ref: `working tree with T115f authority metadata, final rerun evidence, and notes/log closure; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
