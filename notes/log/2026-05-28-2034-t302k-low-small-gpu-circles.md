# 2026-05-28 20:34 T302k Low-Small GPU Circles

## Purpose

Add a GPU-resident low-small semantic component circle helper for the later touchdown keepout loss.

## Stage

`extension/batch_mpc_planner` semantic geometry helper.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'low_small_gpu_circles'
```

Initial failure:

```text
ModuleNotFoundError: No module named 'extension.batch_mpc_planner.semantic_geometry'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'low_small_gpu_circles'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py
```

## Input Conditions

- Baseline ref: `c85a7e5`.
- Semantic id `1` is the low-small obstacle id for this design.
- Helper must keep output tensors on the input device and avoid per-env CPU conversion.

## Key Metrics

- Focused circle tests: `2 passed, 109 deselected`.
- Pycompile: pass.

## Result

Pass locally.

## Conclusion

Added `LowSmallCircles` and `low_small_component_circles()`. The helper uses tensor label propagation over `[B,H,W]`, returns fixed-shape centers/radii/valid/truncated tensors, splits disconnected components in the covered tests, and preserves the semantic map device.

## Follow-Up

Continue with Task 4 touchdown circle keepout loss.

## Git Refs

- Baseline Ref: `c85a7e5`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py](../../Go2Pvcnn/extension/batch_mpc_planner/semantic_geometry.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
