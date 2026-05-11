# T113c Foothold Semantic Policy

## Meta

- Time: `2026-05-07 21:08 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T113c](../todo/T100-batched-together-planner-gpu-migration.md#t113c-small-cross-preference-and-large-bypass-foothold-policy)

## Purpose

- Verify the third execution leaf under `T113`: explicit `small` crossing preference and `large` bypass / forward-refusal semantics at the foothold/touchdown layer.

## Scope

- [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)

## Subagent Result Summary

- `small` remains non-step-on, with bounded comparison among current/front/beyond legal terrain supports.
- The planner may now prefer beyond-small legal terrain when it wins, while still allowing legal terrain before the obstacle when it scores better.
- `large` now triggers foothold-level center-forward refusal by invalidating center-candidate front-leg touchdown support in blocked corridors while preserving lateral bypass candidates when available.
- The implementation stayed pure GPU and fixed shape with no CPU sync or host-side mask branching.

## Verification Commands

1. `python -m py_compile Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`

## Key Metrics

- semantic planner focused suite: `11 passed`
- together guardrail subset: `5 passed`
- core smoke subset: `6 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- `small_surface_touchdown_count == 0`
  - covered through semantic fixtures and touchdown-surface counting helpers
- `large_surface_touchdown_count == 0`
  - covered through the same semantic touchdown-surface assertions
- `F2_small_forward_beyond_better`
  - covered by `test_small_forward_beyond_better_selects_beyond_small_terrain`
- `F3_small_forward_front_better`
  - covered by `test_small_forward_front_better_may_legally_stay_before_obstacle`
- `F4_large_forward_blocking`
  - covered by `test_large_forward_blocking_suppresses_center_candidate_progression`
- `F5_large_both_sides_blocked`
  - covered by `test_large_both_sides_blocked_refuses_unsafe_progression`

## Caveats

- Broader repository `pytest` remains blocked by the known `Go2Pvcnn/tests/conftest.py` dependency on missing `scripts.go2fp`; this leaf stayed on focused together-only verification.
- This leaf proves acceptance behavior through deterministic tests but does not yet add new runtime-visible metrics named `small_cross_preference_outcome` or `large_forward_refusal_ratio`; surfacing remaining acceptance metrics is deferred to `T113e`.

## Conclusion

- `T113c` is complete and verified within the current together-only test path.
- The next recommended execution step is `T113d`, adding height-aware swing clearance and continuous body/thigh/calf collision coverage on top of the new foothold semantics.

## Git Refs

- Baseline Ref: `8e8acc0`
- Candidate Ref: `working tree on top of 8e8acc0 (2026-05-07 21:08 +0800); T113c code/test changes plus notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
