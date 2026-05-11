# T116e Barriers, Direction Guards, And Final Selection

## Purpose

Record the implementation and verification evidence for T116e cost barriers, final candidate selection, command-direction guards, and selected-result diagnostics in the T116 K=5 mode-first small-obstacle crossing rewrite.

## Stage

- Stage: together planner cost/selection implementation
- Related todo: [T100/T116e](../todo/T100-batched-together-planner-gpu-migration.md#t116e-cost-barriers-selection-rules-direction-guards-and-per-leg-diagnostics)
- Baseline Ref: `979b2b5`
- Candidate Ref: working tree on `979b2b5`
- Key files:
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)

## Procedure

- Worker implemented T116e with TDD and local self-review.
- Main agent reproduced focused tests and probed four-direction `CROSS_SMALL` outputs.
- Spec review initially returned blockers:
  - F6 had been weakened to allow `cross_small_success=False`.
  - Base XY overlap was incorrectly treated as failure.
  - Direction guard was only diagnostic.
  - Large semantic and per-leg path hard barriers were incomplete.
  - BYPASS non-center preference could choose hard-barrier candidates.
- Worker fixed these and spec re-review found one remaining blocker:
  - thigh/calf `leg_min_clearance` had been masked by foot-path clearance.
- Worker removed that shortcut and added/adjusted F10 so foot-path clearance cannot substitute for thigh/calf clearance.
- Spec review then approved.
- Quality review found one diagnostic consistency blocker:
  - selected `feasible`/`safe_fallback`/`status` were gathered from raw costs after direction masking.
- Worker fixed selected feasibility/status to gather from direction-masked candidate tensors and added an all-candidates-direction-guard regression.
- Quality re-review approved.

## Commands

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "t116_f2 or t116_f3 or t116_f4 or t116_f6 or t116_f8 or t116_f9 or t116_f10 or t116_f11 or t116_f12 or bypass or direction"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "t116_f6 or t116_f8 or t116_f9 or t116_f10 or cross_small_schedule or path_collision_diagnostic"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q -k "t116 or schema_shape"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q
python -m py_compile Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py
```

## Key Metrics

- T116e/direction/bypass deterministic union: `21 passed, 61 deselected`
- T116 crossing/collision subset: `6 passed, 76 deselected`
- Core T116/schema subset: `3 passed, 5 deselected`
- Guardrail full file: `9 passed`
- `py_compile`: passed with no output
- Four-direction normal small probe:
  - selected mode `CROSS_SMALL`
  - selected beta nonzero
  - `feasible=True`
  - `cross_small_success=True`
  - `command_direction_violation=False`
  - real selected `leg_min_clearance > 0`

## Result

Pass.

T116e now rejects unsafe candidates with hard barriers, applies direction guard before selection, keeps selected status coherent with masked candidate costs, and surfaces selected diagnostics needed by T116 runtime output and final tests.

## Conclusion

- Normal crossable small obstacles now have a positive four-direction deterministic acceptance fixture.
- Body/base/foot/thigh/calf safety is barrier-backed.
- Thigh/calf clearance remains a real kinematic/terrain criterion; foot-path clearance is additional evidence, not a substitute.
- Base overlap with a small obstacle footprint is diagnostic unless clearance/penetration fails.
- Large and too-high small obstacle cases use BYPASS mode with coherent non-center/center behavior depending on feasible non-center availability.
- Direction guard cannot silently report OK status after masking out all candidates.

## Follow-Up

- Continue with T116f deterministic and guardrail cleanup.
- Do not treat focused T116e passes as final authority; T116h must rerun semantic/core/guardrail and runtime checks together on the final code state.

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: working tree verification on `979b2b5`
- Current Work Ref: uncommitted working tree on `979b2b5`
