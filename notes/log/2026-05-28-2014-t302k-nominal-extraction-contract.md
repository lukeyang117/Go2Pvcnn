# 2026-05-28 20:14 T302k Nominal Extraction Contract

## Purpose

Restore the parametric MPC contract where semantic-aware nominal construction is outside `decode_parametric_trajectory()`, so later low-small loss work can optimize trajectories without decode-time semantic repair.

## Stage

`extension/batch_mpc_planner` parametric MPC nominal/decode/planner boundary.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red contract test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q -k 'nominal or consumes_nominal'
```

Initial failure:

```text
ImportError: cannot import name 'build_parametric_nominal'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or semantic'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/extension/batch_mpc_planner/planner.py
```

## Input Conditions

- Baseline ref: `97c5b60`.
- Current code before this task had `decode_parametric_trajectory(state, terrain, command, variables, horizon=...)`.
- High/large root bypass logic was still inside `parametric.py`.
- No `ParametricTrajectoryNominal` object existed in `semantic_policy.py`.

## Key Metrics

- Parametric tests: `15 passed`.
- Backend focused tests: `30 passed, 76 deselected, 1 warning`.
- Pycompile: pass.
- Intermediate regression found and fixed: pure-yaw high/large semantic policy initially failed with root min distance `0.2027m` versus required `>=0.305m`; after adding yaw-active semantic candidate handling, the focused backend subset passed.

## Result

Pass locally.

## Conclusion

`semantic_policy.py` now owns `ParametricTrajectoryNominal` and `build_parametric_nominal()`. `planner.py` builds the nominal once before optimization. `decode_parametric_trajectory()` consumes `nominal + variables` and no longer performs high/large semantic search internally.

## Follow-Up

- Commit Task 1.
- Continue with Task 2 terrain metadata before implementing the confirmed low-small losses.

## Git Refs

- Baseline Ref: `97c5b60`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py](../../Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
