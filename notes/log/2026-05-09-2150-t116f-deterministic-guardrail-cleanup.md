# T116f Deterministic And Guardrail Cleanup

## Purpose

Record the final deterministic/guardrail cleanup pass for the T116 `K=5` mode-first small-obstacle crossing rewrite.

## Stage

- Stage: together planner deterministic tests and guardrails
- Related todo: [T100/T116f](../todo/T100-batched-together-planner-gpu-migration.md#t116f-deterministic-and-guardrail-test-rewrite-on-final-deterministic-code-state)
- Baseline Ref: `979b2b5`
- Candidate Ref: working tree on `979b2b5`
- Key files:
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)

## Procedure

- Rewrote or downgraded old active deterministic expectations so `T113-T115` remain historical evidence rather than current behavior authority.
- Added guardrails against active old `front_cross/rear_follow/clear` state contracts in the hot path.
- Fixed a production route-offset regression found during T116f review: BYPASS route ids were correct, but `route_offset_m` was still zero, so LEFT/RIGHT candidates did not affect root/support/foothold geometry.
- Restored `foot_large_collision_count > 0` as a hard barrier after review caught that weakening it would violate the body/feet/leg obstacle-safety contract.
- Corrected the large-obstacle deterministic fixture from an initial-foot-overlap case to a forward large-obstacle candidate-route case.
- Renamed the F12 large-bypass test to make its scope precise: it now proves a non-hard-barrier non-center route candidate is selected, not final IsaacLab runtime acceptance.
- Ran spec compliance review and code-quality review; the first code-quality review raised the F12 overclaim, and re-review approved after the test was renamed and narrowed.

## Commands

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "t116_f12_large_blocks_center or t116_bypass_center_zero_not_successful_when_safe_non_center_exists or t116_f11_too_high_small_uses_bypass"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q
python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py
```

## Key Metrics

- Focused large/too-high deterministic subset: `5 passed, 30 deselected`
- Full deterministic/core/guardrail union: `52 passed`
- Guardrail full file: `9 passed`
- `py_compile`: passed with no output
- Spec compliance review: `APPROVED`
- Code-quality re-review after F12 scope correction: `APPROVED`

## Result

Pass for T116f.

## Conclusion

- T116 deterministic tests no longer certify old `K=3`, `35`-frame, or `front_cross/rear_follow/clear` behavior as current architecture.
- Production hot path no longer keeps the old post-rollout state classifier contracts used by T113-T115.
- BYPASS route offsets now enter rollout/support/foothold generation, so command-relative LEFT/RIGHT routes are not only labels on candidate commands.
- Large-obstacle deterministic coverage now keeps obstacle collision hard barriers intact and proves non-center route construction without claiming final runtime feasibility.

## Follow-Up

- T116g must validate `env_isaacsim` headless runtime behavior, including large-bypass `feasible/status`, base path, feet path, touchdowns, body/thigh/calf collision, and four command directions.
- T116h must rerun deterministic, guardrail, and bounded runtime checks together on the final code state.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: working tree verification on `979b2b5`
- Current Work Ref: uncommitted working tree on `979b2b5`
