# 2026-05-28 20:48 T302k Touchdown Circle Keepout

## Purpose

Replace the sampled parametric low-small crossing loss key with a touchdown circle keepout loss that only triggers for touchdowns sampled on low-small semantic cells.

## Stage

`extension/batch_mpc_planner` parametric sampled losses.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo/T302k-low-small-loss-redesign-plan.md](../todo/T302k-low-small-loss-redesign-plan.md)

## Command / Procedure

Red test:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'touchdown_keepout'
```

Initial failure:

```text
ModuleNotFoundError: No module named 'extension.batch_mpc_planner.parametric_losses'
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'touchdown_keepout or exposes_sampled_frame_losses'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/config.py
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or semantic'
```

## Input Conditions

- Baseline ref: `6ee1962`.
- Low-small semantic id is `1`.
- No hard projection or touchdown snapping is allowed.
- Old `low_small_crossing` config remains only as a shared threshold source for semantic classification and existing standalone progress-loss tests; it is no longer a sampled parametric optimizer loss key.

## Key Metrics

- Focused keepout/sample-loss tests: `4 passed, 109 deselected`.
- Backend parametric/semantic subset: `32 passed, 81 deselected, 1 warning`.
- Pycompile: pass.

## Result

Pass locally.

## Conclusion

Added `parametric_touchdown_keepout_loss()` in `parametric_losses.py`, wired `cfg.losses.touchdown_keepout`, and replaced sampled loss key `parametric_low_small_crossing` with `parametric_touchdown_keepout`.

## Follow-Up

Continue with Task 5 swing target terrain clearance loss.

## Git Refs

- Baseline Ref: `6ee1962`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
