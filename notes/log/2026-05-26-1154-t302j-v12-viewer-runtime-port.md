# T302j V12 Viewer Runtime Port

## Purpose

Make the V12 touchdown farthest-export behavior available in the actual viewer/runtime path so visual inspection can use the same behavior as the probe.

## Stage

- `extension/viz/go2_foostep_planner.py`
- `extension/batch_mpc_planner`

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)

## Procedure

Updated viewer debug-variant help text to point at `reachable_fk_cross_v12` and added a focused test that the viewer MPC cfg forwards `reachable_fk_cross_v12` into the runtime cfg without pre-applying command-specific weights.

Smoke command:

```bash
timeout 90s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --planner-backend mpc \
  --mpc-debug-variant reachable_fk_cross_v12 \
  --n-frames 25 \
  --scripted-command '0.50 0.25 1.00' \
  --scripted-command-cycles 1 \
  --warmup-steps 2 \
  --headless \
  > tmp/t302i-viewer-realized-foot-mismatch/t302j_v12_viewer_smoke.log 2>&1
```

## Result

Viewer startup reached:

- `[Viewer] Attached mpc trajectory manager`
- `[Viewer] Planner horizon: 25 frames @ dt=0.020s`
- `[Viewer] Playback mode: kinematic (no physics)`
- `[Viewer][Playback] path=render+scene_sync`

The smoke was stopped manually after confirming the startup/runtime path to avoid occupying the user's visual session.

## Verification

```bash
pytest -q \
  Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_build_mpc_cfg_forwards_debug_variant_name_without_preapplying_weights \
  Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_main_builds_only_selected_backend_planner_cfgs \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_debug_v12_touchdown_export_uses_command_farthest_swing_point
```

Result: `3 passed`.

```bash
python -m py_compile \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/test_viewer_reset.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py
```

Result: pass.

## Conclusion

The V12 behavior is now selectable from the viewer with:

```bash
--planner-backend mpc --mpc-debug-variant reachable_fk_cross_v12
```

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree @ c54dc5c plus V12 viewer runtime port`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
