# T114g Traceability And Authority

## Meta

- Time: `2026-05-08 17:05 +0800`
- Stage: `together planner focused implementation verification`
- Result: `pass`
- Todo: [T100/T114g](../todo/T100-batched-together-planner-gpu-migration.md#t114g-deterministic-metricfixture-traceability-and-final-rerun-authority)

## Purpose

- Verify the final execution leaf under `T114`: deterministic fixture/metric traceability and enforceable final-code-state rerun authority.

## Scope

- [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
- [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
- [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)

## TDD Evidence

- RED verification first:
  - `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114g_'`
  - result: `4 failed, 35 deselected`
  - failing reason: missing executable traceability/authority surface (`T114_DETERMINISTIC_FIXTURE_TO_TESTS`, `T114_REQUIRED_METRIC_TO_TESTS`, `T114_FINAL_AUTHORITY_RECORD`)
- GREEN after minimal implementation:
  - the same `t114g_` subset rerun passed

## Verification Commands

1. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k 't114g_'`
2. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q`
3. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py -q`
4. `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
5. `python -m py_compile Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py`

## Key Metrics

- TDD red subset: `4 failed`
- focused `T114g` subset after implementation: `4 passed`
- semantic authoritative union: `39 passed`
- core authoritative union: `6 passed`
- guardrail authoritative union: `5 passed`
- `py_compile`: `pass`

## Acceptance Coverage

- fixtures `F1`, `F1b`, and `F2`-`F15` are explicitly and completely mapped in executable test metadata
- required section-12.2 metrics are explicitly and completely mapped in executable test metadata
- guardrail/static checks are explicitly included in the authoritative final rerun surface
- older overlapping focused subsets are explicitly marked `superseded / non-authoritative`
- the authoritative final-code-state rerun surface is explicit and executable at the test layer

## Caveats

- The authority contract is implemented as executable test metadata in the semantic planner test surface, not as a repository-wide external harness.
- This leaf intentionally avoided production behavior changes and closed traceability/authority at the test layer.
- Broader repository-wide `pytest` remains outside this leaf’s scope and still inherits the known repo-level raw dependency caveat.

## Conclusion

- `T114g` is complete and verified.
- `T114` is now complete at the focused authoritative-union level.

## Git Refs

- Baseline Ref: `current working tree after T114f verification`
- Candidate Ref: `working tree with T114g traceability/authority closure and focused verification; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py](../../Go2Pvcnn/tests/test_batched_together_semantic_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [index.md](index.md)
