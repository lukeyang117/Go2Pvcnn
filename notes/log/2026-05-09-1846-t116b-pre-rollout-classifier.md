# T116b Pre-Rollout Classifier And GPU Small Geometry

## Purpose

Record T116b implementation and verification for the K5 mode-first together planner rewrite.

## Stage

- Stage: `Go2Pvcnn/extension/batched_together_planner` classifier / semantic geometry
- Related todo: [T100/T116b](../todo/T100-batched-together-planner-gpu-migration.md#t116b-pre-rollout-mode-classifier-and-gpu-small-geometry)

## Procedure

- Worker implemented `classify_mode_and_geometry(...)` and T116 mode constants in [parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py).
- First spec review required fixes for large-only bypass priority and separated-small geometry.
- Re-review required a narrower fix for too-high-small `all_clear` priority.
- Main agent reran focused tests and py-compile after the fixes.
- Final spec re-review approved.
- Code quality review approved with only non-blocking follow-ups.

## Commands

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "t116_f5 or t116_f7 or pre_rollout_classifier or classifier_large_only or separated_small or too_high"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q -k "old_front_rear_clear or no_cpu_sync or t116"
python -m py_compile Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_batched_together_guardrails.py
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "large"
```

## Key Metrics

- T116b semantic focused: `6 passed, 60 deselected`
- T116b guardrail focused: `5 passed, 4 deselected`
- `py_compile`: pass
- Broad historical `-k large`: `3 failed, 2 passed, 60 deselected`
- Spec re-review: `APPROVED`
- Quality review: `APPROVED`

## Result

- `T116_MODE_CRUISE`, `T116_MODE_APPROACH_SMALL`, `T116_MODE_CROSS_SMALL`, and `T116_MODE_BYPASS_OBSTACLE` are now the T116 classifier mode set.
- `large` in the command corridor and too-high `small` keep bypass priority over any `all_clear` branch.
- `all_clear` is only true when a small obstacle is present and body rear plus all four foot anchors have cleared the selected small back edge.
- Separated small obstacles no longer collapse into one huge front/back envelope in the classifier; the implementation uses a fixed-shape nearest envelope.
- Hot-path guardrails cover old mode names and CPU/dynamic extraction in `classify_mode_and_geometry`.

## Residual Risk

- The nearest-small envelope is a fixed-window heuristic rather than true connected-component extraction; quality review accepted it for current T116b and the current small-obstacle footprint.
- `planner.py` does not consume `classify_mode_and_geometry(...)` yet. T116c owns mode table gathering and candidate expansion.
- Broad `-k large` still includes old T114/T113 route/state tests expecting pre-T116 contracts such as `semantic_candidate_count`, `STATE_BYPASS`, or old lateral offset behavior. This is logged as T116f cleanup, not a T116b classifier blocker.

## Follow-Up

- Start T116c: consume pre-rollout `mode_code` in candidate table gathering and fixed `[B,5] -> [B*5]` expansion.
- Keep runtime `env_isaacsim` tests deferred until T116g.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: working tree on top of `130c635`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
