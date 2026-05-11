# T116d Dynamic CROSS_SMALL Gait Generation

## Purpose

Record T116d implementation and verification for dynamic `CROSS_SMALL` schedule, touchdown, apex, and foot-path diagnostics.

## Stage

- Stage: `Go2Pvcnn/extension/batched_together_planner` crossing gait generation
- Related todo: [T100/T116d](../todo/T100-batched-together-planner-gpu-migration.md#t116d-dynamic-cross_small-touchdownapexfoot-path-generation)

## Procedure

- Worker added TDD coverage for F6/F8/F9 and explicit command-leading schedule ordering.
- Worker implemented `build_cross_small_schedule(...)`, CROSS_SMALL touchdown/apex generation, and per-leg path diagnostics.
- Spec review approved fixed-shape `torch.argsort` over four legs and confirmed no T116e overreach.
- Quality review found a blocker: foot-path diagnostics were self-masking collisions by lifting sampled z before measuring clearance.
- Worker added a low-path collision regression and fixed diagnostics to use actual sampled path z.
- Main agent reran focused tests, guardrails, and py-compile.
- Quality re-review approved.

## Commands

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "t116_f6 or t116_f8 or t116_f9 or cross_small_schedule or path_collision"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "t116_f1 or t116_f3 or t116_f4 or t116_f11 or t116_f12"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q
python -m py_compile Go2Pvcnn/extension/batched_together_planner/schedule.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py
```

## Key Metrics

- T116d focused after blocker fix: `6 passed, 70 deselected`
- T116c regression focused: `5 passed, 71 deselected`
- Guardrail full file: `9 passed`
- `py_compile`: pass
- Spec review: `APPROVED`
- Quality review: `CHANGES_REQUIRED` then `APPROVED`

## Result

- CROSS_SMALL candidates use a command-relative schedule with command-leading legs before trailing legs.
- CROSS_SMALL touchdowns are generated beyond selected `small_back_s + margin`.
- Touchdowns are checked against semantic terrain and exposed through per-leg diagnostics.
- Swing apex is raised above `small_top_z` with clearance/robot margins.
- Foot-path diagnostics now compute clearance from actual sampled path z, not an auto-lifted path.
- A low-path regression proves semantic-small path collisions can be reported.
- `cross_small_success` now depends on selected CROSS_SMALL mode, beyond-back-edge touchdowns, zero touchdown-on-small, zero foot-path-small collision, and valid command-leading schedule.

## Residual Risk

- Full body/thigh/calf barrier selection remains T116e scope.
- Cost barriers and final candidate selection still remain T116e scope; this leaf only generates and surfaces diagnostics.
- Runtime Isaac Lab validation remains deferred until T116g.

## Follow-Up

- Start T116e: turn generated diagnostics into selection barriers, direction guards, and final per-leg/body/leg safety decisions.
- T116f must keep the low-path collision regression when cleaning old K=3/front-rear tests.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: working tree on top of `130c635`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/schedule.py](../../Go2Pvcnn/extension/batched_together_planner/schedule.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
