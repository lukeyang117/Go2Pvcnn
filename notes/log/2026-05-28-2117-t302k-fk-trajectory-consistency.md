# 2026-05-28 21:17 T302k FK Trajectory Consistency

## Purpose

Add optimized-target versus FK-realized foot trajectory consistency to expose absolute and root-relative mismatch.

## Stage

`extension/batch_mpc_planner` final parametric loss breakdown.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'trajectory_consistency'
```

Initial failure:

```text
ImportError: cannot import name 'parametric_trajectory_fk_consistency_loss'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'trajectory_consistency'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or semantic'
```

## Input Conditions

- Baseline ref: `7d2b32f`.
- User specified no extra parameter for this loss.

## Key Metrics

- Focused trajectory consistency test: `1 passed, 116 deselected`.
- Backend parametric/semantic subset: `32 passed, 85 deselected, 1 warning`.
- Pycompile: pass.

## Result

Pass locally.

## Conclusion

Added `parametric_trajectory_fk_consistency_loss()` and final loss key `parametric_trajectory_fk_consistency`.

## Follow-Up

Continue with Task 8 plane root-z target loss.

## Git Refs

- Baseline Ref: `7d2b32f`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
