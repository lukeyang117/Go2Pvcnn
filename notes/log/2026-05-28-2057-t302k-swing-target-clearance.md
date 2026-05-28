# 2026-05-28 20:57 T302k Swing Target Clearance

## Purpose

Add a parametric swing target terrain clearance loss that checks all target swing foot points against terrain height, without separating semantic and terrain clearance.

## Stage

`extension/batch_mpc_planner` parametric sampled losses.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'swing_target_clearance or sampled_frame_losses'
```

Initial failure:

```text
ImportError: cannot import name 'parametric_swing_foot_clearance_loss'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'swing_target_clearance or sampled_frame_losses'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/config.py
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or semantic'
```

## Input Conditions

- Baseline ref: `b018d38`.
- Confirmed parameter: `swing_foot_clearance_margin_m`, initial value `0.02`.

## Key Metrics

- Focused swing/sample-loss tests: `2 passed, 112 deselected`.
- Backend parametric/semantic subset: `32 passed, 82 deselected, 1 warning`.
- Pycompile: pass.

## Result

Pass locally.

## Conclusion

Added `parametric_swing_foot_clearance_loss()` and sampled loss key `parametric_swing_foot_clearance`, weighted by `cfg.losses.swing_foot_clearance`.

## Follow-Up

Continue with Task 6 FK body/leg terrain collision loss.

## Git Refs

- Baseline Ref: `b018d38`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
