# T116c Mode Candidate Tables And Command-Relative Routes

## Purpose

Record T116c implementation and verification for fixed K=5 mode candidate tables, command-relative routes, and `[B,5] -> [B*5]` expansion.

## Stage

- Stage: `Go2Pvcnn/extension/batched_together_planner` top-level candidate expansion
- Related todo: [T100/T116c](../todo/T100-batched-together-planner-gpu-migration.md#t116c-mode-candidate-tables-command-relative-routes-and-k5-candidate-expansion)

## Procedure

- Worker added TDD coverage for T116 F1/F3/F4/F11/F12.
- Worker consumed `classify_mode_and_geometry(...)` in [planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py) before rollout.
- Worker added fixed K=5 beta/route tables and command-relative candidate command expansion.
- Main agent reran focused tests, full guardrails, and py-compile.
- Spec review approved.
- Quality review approved with two non-blocking follow-ups.

## Commands

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "t116_f1 or t116_f3 or t116_f4 or t116_f11 or t116_f12"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q -k "t116_f13 or schema_shape or k5"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q
python -m py_compile Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py
```

## Key Metrics

- T116c semantic focused: `5 passed, 66 deselected`
- T116 core focused: `2 passed, 6 deselected`
- Guardrail full file: `9 passed`
- `py_compile`: pass
- Spec review: `APPROVED`
- Quality review: `APPROVED`

## Result

- `plan_segment(...)` now classifies global mode before candidate rollout.
- Candidate tables are gathered by `mode_code [B]` and expanded through fixed `[B,5] -> [B*5]` tensors.
- CRUISE uses beta `[1.0, 0.75, 0.5, 0.25, 0.0]` with center routes.
- APPROACH_SMALL uses beta `[0.8, 0.6, 0.4, 0.2, 0.0]` with center routes.
- CROSS_SMALL uses beta `[0.5, 0.35, 0.2, 0.1, 0.0]` with center routes in this leaf.
- BYPASS_OBSTACLE uses beta `[0.5, 0.25, 0.5, 0.25, 0.0]` with `[LEFT, LEFT, RIGHT, RIGHT, CENTER]`.
- Route motion is command-relative through candidate commands. The categorical `selected_route` is the T116 route identity.

## Residual Risk

- `selected_route_offset` remains zero as a compatibility field; it is not the authoritative T116 route diagnostic.
- Some tests use private helper assertions for table contents. T116f should add black-box `plan_segment` assertions that fail if planner stops consuming the tables.
- T116e still owns final bypass/center-route barriers and selection rules. Current costs may select the zero-center candidate in bypass mode until those barriers exist.

## Follow-Up

- Start T116d: dynamic `CROSS_SMALL` schedule/touchdown/apex/foot-path generation.
- Keep runtime `env_isaacsim` tests deferred until T116g.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: working tree on top of `130c635`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
