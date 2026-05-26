# T302k Parametric Default FK Output

## Purpose

Verify the first executable parametric MPC contract in `extension/batch_mpc_planner`: default `plan_segment` should decode parametric root/foot curves, solve clamped IK, and export FK-realized feet instead of dense Cartesian target feet.

## Stage

`extension/batch_mpc_planner` planner default contract.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Procedure

TDD sequence:

1. Added `test_parametric_plan_exports_fk_realized_feet`.
2. Confirmed red failure: `result.foot_pos` differed from FK output, max z mismatch about `0.126m`.
3. Added `MpcRuntimeCfg.use_parametric_trajectory` and a parametric `plan_segment` branch.
4. Added default-path test `test_plan_segment_defaults_to_parametric_fk_realized_feet`.
5. Confirmed red failure before switching the default.
6. Switched default `use_parametric_trajectory=True`.
7. Updated old dense optimizer behavior tests to opt into `use_parametric_trajectory=False`.

## Verification

Commands:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k parametric_plan_exports_fk_realized_feet
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "defaults_to_parametric_fk_realized_feet or parametric_plan_exports_fk_realized_feet"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q -k "parametric or zero_command or output"
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Results:

- First targeted red before fix: `1 failed`; max FK mismatch `0.125999987m`.
- Default-path red before default switch: `1 failed`; same FK mismatch class.
- Targeted green after implementation: `2 passed, 112 deselected`.
- Focused subset: `15 passed, 107 deselected`.
- Backend + parametric suite: `122 passed, 1 warning`.

The warning is an existing PyTorch scalar conversion warning in `test_mpc_semantic_contact_avoidance_loss_has_xy_gradient_from_soft_field`.

## Result

Pass for T302k.1-T302k.4 local/backend contract:

- Parametric helpers and decode exist in `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`.
- `plan_segment` defaults to parametric trajectory decode.
- Exported nonzero-command `foot_pos` equals FK from exported `joint_angles`.
- Touchdown z remains grounded from `height_at`.
- The old dense optimizer route remains available only via `mpc_use_parametric_trajectory=False`.

## Follow-Up

T302k.5 remains open: replace placeholder `parametric_target_fk_error` diagnostics with sampled 25-frame losses for reachability, terrain/semantic collision, gait/timing, command progress, and curve regularization.

## Git Refs

- Baseline Ref: `1b799cd`
- Candidate Ref: working tree on top of `1b799cd`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
