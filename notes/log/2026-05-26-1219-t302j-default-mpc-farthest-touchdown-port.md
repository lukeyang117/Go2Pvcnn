# T302j Default MPC Farthest Touchdown Port

## Purpose

Correct the prior V12 debug-only integration. The user requires the farthest-touchdown export behavior to be part of the normal MPC runtime path, not gated by `--mpc-debug-variant reachable_fk_cross_v12`, `--n-frames 25`, or scripted command flags.

## Stage

- `extension/batch_mpc_planner`
- `extension/viz`

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)

## Change

- Moved command-direction farthest touchdown export into the default `plan_segment()` MPC export path.
- Removed the requirement that `cfg.debug_loss_variant == "reachable_fk_cross_v12"` for this export behavior.
- Reverted viewer help text away from suggesting V12 as the normal way to enable the fix.
- Removed the viewer test that treated V12 debug variant as the required runtime trigger.
- Added a default `plan_segment()` test proving farthest touchdown export happens without debug variant.

## Verification

```bash
pytest -q \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_debug_v12_touchdown_export_uses_command_farthest_swing_point \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_plan_segment_exports_command_farthest_touchdown_by_default
```

Result: `2 passed`.

```bash
pytest -q Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_main_builds_only_selected_backend_planner_cfgs
```

Result: `1 passed`.

```bash
python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_viewer_reset.py
```

Result: pass.

User-command smoke, without debug variant:

```bash
timeout 90s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:1 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc \
  --terrain-row 6 \
  > tmp/t302i-viewer-realized-foot-mismatch/t302j_default_mpc_viewer_smoke_user_cmd.log 2>&1
```

Reached:

- `[Viewer] Attached mpc trajectory manager`
- `[Viewer] Planner horizon: 50 frames @ dt=0.020s`
- `[Viewer] Terrain tile override: row=6 col=0 origin=(+12.000, -76.000, +0.000)`
- `[Viewer][Playback] path=render+scene_sync`

The smoke was stopped manually after startup confirmation to avoid occupying the user's livestream session.

## Conclusion

The user's original command now uses the farthest-touchdown export behavior by default because it runs `--planner-backend mpc`. No V12 debug flag, scripted command, or `n-frames=25` override is needed.

## Git Refs

- Baseline Ref: `working tree @ c54dc5c`
- Candidate Ref: `working tree @ c54dc5c plus default MPC farthest-touchdown export`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
