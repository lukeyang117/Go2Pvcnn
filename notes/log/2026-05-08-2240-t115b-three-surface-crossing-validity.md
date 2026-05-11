# T115b Three-Surface Crossing Validity

## Meta

- Time: `2026-05-08 22:40 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T115b](../todo/T100-batched-together-planner-gpu-migration.md#t115b-touchdown--feet--base-three-surface-crossing-validity)

## Purpose

- Implement the deterministic/core portion of `T115b` only.
- Split crossing validity into three explicit surfaces:
  - touchdown surface
  - feet path surface
  - base path surface
- Make these surfaces both observable and enforceable in planner/core selection without touching the runtime harness leaves.

## Scope

- Code changed only in:
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
- No runtime harness, viewer diagnostics, or Isaac Lab runtime acceptance files were modified in this leaf.

## Implementation Notes

- Added rollout/result diagnostics:
  - `touchdown_on_small_count`
  - `front_foot_small_collision_count`
  - `rear_foot_small_collision_count`
  - `front_foot_min_clearance_to_small`
  - `rear_foot_min_clearance_to_small`
  - `base_small_penetration_count`
  - `base_min_clearance_to_small`
  - `base_path_crosses_small_flag`
- Exposed these metrics on both:
  - `TogetherRollout`
  - `TogetherPlannerResult`
- Added candidate-level planner gating:
  - crossing-state candidates are infeasible if touchdown lands on `small`
  - crossing-state candidates are infeasible if front or rear foot swing collides with `small`
  - crossing-state candidates are infeasible if base/body path penetrates or crosses `small`
- Added deterministic `small-strip` fixture coverage that forces planner-side three-surface invalidation without introducing a runtime harness dependency.
- Normalized no-contact/no-overlap clearance surfaces to finite large-positive diagnostics so selected planner outputs stay numerically usable in deterministic tests.

## Verification

### TDD red

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't115b_'`
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q -k 'schema_shape_and_device'`
- Initial result before implementation:
  - `3 failed`
  - `1 failed`
- Initial failure surface:
  - missing three-surface rollout/result metrics
  - missing schema exposure on `TogetherPlannerResult`

### Focused T115b

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't115b_'`
- Result:
  - `3 passed`

### Affected semantic union

- Command:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
- Result:
  - `48 passed`

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

- touchdown surface:
  - `touchdown_on_small_count == 0` is required for crossing-state feasibility
- feet path surface:
  - `front_foot_small_collision_count == 0` is required for crossing-state feasibility
  - `rear_foot_small_collision_count == 0` is required for crossing-state feasibility
  - min-clearance diagnostics are exposed via `front_foot_min_clearance_to_small` and `rear_foot_min_clearance_to_small`
- base path surface:
  - `base_small_penetration_count == 0` is required for crossing-state feasibility
  - `base_path_crosses_small_flag == 0` is required for crossing-state feasibility
  - min-clearance diagnostic is exposed via `base_min_clearance_to_small`
- successful crossing candidates now fail planner/core feasibility if any of the three surfaces violate the `small`-crossing contract

## Conclusion

- `T115b` deterministic/core scope is implemented and verified.
- Three-surface crossing validity now participates in real planner feasibility gating, not only diagnostics surfacing.
- `T115d/T115e` remain open for the real `env_isaacsim` headless runtime path.

## Follow-up

- Continue with `T115d/T115e` when the runtime harness leaf is active.
- Reuse the new planner/core metrics directly in runtime diagnostics rather than inventing a parallel metric surface.

## Git Refs

- Baseline Ref: `working tree after T115a deterministic/core grounded-contract patch`
- Candidate Ref: `working tree with T115b three-surface crossing diagnostics + feasibility gating`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
