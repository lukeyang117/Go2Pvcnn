# T114e Anchor-To-Touchdown Path Clearance

## Meta

- Time: `2026-05-08 16:06 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T114e](../todo/T100-batched-together-planner-gpu-migration.md#t114e-anchor-to-touchdown-footleg-path-clearance)

## Purpose

- Verify the fifth execution leaf under `T114`: candidate-stage anchor-to-touchdown foot/leg path-clearance diagnostics and invalidation.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
- [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114e_'`
  - result: `3 failed, 28 deselected`
  - failing reason: missing `anchor_to_touchdown_foot_clearance`, `anchor_to_touchdown_leg_clearance`, and `candidate_path_collision_flag` on `TogetherRollout`
- GREEN after minimal implementation:
  - the same `t114e_` subset rerun passed

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114e_'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
5. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`

## Key Metrics

- TDD red subset: `3 failed`
- focused `T114e` subset after implementation: `3 passed`
- semantic affected-union rerun: `31 passed`
- core affected-union rerun: `6 passed`
- guardrail rerun: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `anchor_to_touchdown_foot_clearance`, `anchor_to_touchdown_leg_clearance`, and `candidate_path_collision_flag` are now directly test-observable
- endpoint-legal but transition-bad candidate invalidation is covered by a deterministic fixture
- mild path-clearance penalty while remaining feasible is covered by a deterministic fixture
- candidate-stage path-clearance penalty is explicit as `J_path_clearance`
- guardrail suite remained green, preserving pure-GPU/no-`for` hot-path constraints

## Caveats

- The new path-clearance diagnostics currently live on `TogetherRollout`, not yet on `TogetherPlannerResult`; that matches this leaf’s scope, but later result-surface work may surface them outward if needed.
- Path-clearance thresholds and weights remain heuristic tuning knobs.
- In the current deterministic bad-path fixtures, leg-path clearance is the stronger discriminator than foot-path clearance, which is acceptable for this leaf but may warrant later tuning.

## Conclusion

- `T114e` is complete and verified with TDD evidence plus final affected-union reruns on the final code state.
- Recommended next leaf is `T114f`.

## Git Refs

- Baseline Ref: `current working tree after T114d verification`
- Candidate Ref: `working tree with T114e path-clearance changes and focused verification; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
