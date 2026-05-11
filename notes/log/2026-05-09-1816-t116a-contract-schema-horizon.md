# T116a Contract Schema Horizon

## Purpose

Record T116a implementation, review, and verification for the `K=5` / 50-step public contract rewrite.

## Stage

`Go2Pvcnn/extension/batched_together_planner` contract/schema/horizon implementation.

## Related Todo

- [T100/T116a](../todo/T100-batched-together-planner-gpu-migration.md#t116a-contractschemahorizonk5-rewrite)

## Procedure

- Dispatched worker James for T116a TDD implementation.
- Worker confirmed red tests for old 35-step / K=3 / missing diagnostics behavior.
- Main agent reviewed diff, narrowed `planner.py` after spec review found selection overreach, and reran focused tests.
- Spec reviewer Hilbert approved after `planner.py` became interface-only.
- Quality reviewer Arendt required `selected_route` categorical semantics, viewer diagnostic passthrough, and early together `--n-frames=50` validation.
- Worker fixed quality findings with red/green tests.
- Main agent reran focused verification and closed subagents.

## Commands

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q -k "t116_f13 or t116_f14 or schema_shape"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q -k "t116_candidate_axis or t116_horizon"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viz_playback.py -q -k "t116 or together_n_frames or adapt_together"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q
python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/manager.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py Go2Pvcnn/tests/test_viz_playback.py
```

## Results

- Core focused: `3 passed, 5 deselected`
- Guardrail focused: `2 passed, 5 deselected`
- Viewer T116 focused: `2 passed, 33 deselected`
- Core file: `8 passed`
- Guardrail file: `7 passed`
- `py_compile`: pass
- Spec review: approved
- Quality review: approved after fixes

## Key Metrics

- `candidate_count = 5`
- `FIXED_HORIZON_S = 1.0`
- `FIXED_HORIZON_STEPS = 50`
- `FIXED_EVENT_CAP = 2`
- `semantic_candidate_costs` shape `[B,5]`
- `selected_route` is categorical route id, placeholder `CENTER=0`
- Viewer adapter carries T116 placeholder diagnostics

## Conclusion

T116a is complete as an interface-only leaf. It intentionally does not implement pre-rollout mode classification, mode tables, crossing gait/touchdown generation, or final cost barriers. Those remain owned by T116b-e.

## Follow-Up

- Start T116b: pre-rollout mode classifier and GPU small-obstacle geometry.
- Do not treat T116a focused passes as final T116 authority; T116h must rerun deterministic, guardrail, viewer/runtime unions on the final code state.

## Git Refs

- Baseline Ref: working tree on top of `7cf6c11`
- Candidate Ref: uncommitted T116a working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/manager.py](../../Go2Pvcnn/extension/batched_together_planner/manager.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
  - [../../Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
