# T302i V9 Viewer Runtime Port

## Purpose

Expose the probe-only `reachable_fk_cross_v9` loss behavior through the real MPC viewer/runtime path so the user can visually inspect the same V9 behavior.

## Stage

- `extension/batch_mpc_planner`
- `extension/viz/go2_foostep_planner.py`
- T302i low-small reachable crossing diagnostics

## Related Todo

- [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Commands

```bash
pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_build_mpc_cfg_applies_debug_variant_v9 Go2Pvcnn/tests/test_viewer_reset.py::test_viewer_main_builds_only_selected_backend_planner_cfgs

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --commands 'mixed_yaw_v050:0.50 0.25 1.00' --variants baseline,reachable_fk_cross_v9 --cycles 1 --requested-n-frames 300 --warmup-steps 6 > tmp/t302i-viewer-realized-foot-mismatch/reachable_low_small_mixed_yaw_fk_cross_v9_runtime_recheck.jsonl 2>&1

timeout -s INT -k 20s 120s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc --n-frames 25 --plan-dt 0.02 --warmup-steps 6 --scripted-command '0.50 0.25 1.00' --scripted-command-cycles 1 --mpc-debug-variant reachable_fk_cross_v9 > tmp/t302i-viewer-realized-foot-mismatch/viewer_mpc_low_small_v9_headless_smoke.log 2>&1
```

## Key Metrics

- Local tests: `29 passed`.
- Viewer smoke now reaches:
  - `[Viewer] Attached mpc trajectory manager`
  - `Planner horizon: 25 frames @ dt=0.020s`
  - `[Viewer][Playback] path=render+scene_sync`
- Mixed-yaw runtime probe, baseline -> V9:
  - `command_direction_cosine`: `-0.595472 -> 0.998519`
  - `lateral_drift_m`: `0.399854 -> 0.069376`
  - `terminal_planned_vs_fk_foot_error_max`: `0.388287 -> 0.353239`
  - `touchdown_ik_fk_error_max`: `0.661772 -> 0.745050`
  - `raw_ik_joint_limit_violation_max`: `2.039664 -> 1.683324`
  - `fk_swing_foot_step_max_to_median`: `12.329788 -> 6.369946`
  - `fk_swing_foot_accel_max_to_mean`: `11.311711 -> 7.237632`
  - `root_height_min`: `0.289896 -> 0.117291`
  - stance/touchdown/small penetration rates: all `0`.

## Result

Partial pass for visualization reproduction:

- V9 is now selectable in the real viewer with `--mpc-debug-variant reachable_fk_cross_v9`.
- The viewer no longer constructs the fixed-horizon together config for `--planner-backend mpc --n-frames 25`; the previous guard is cleared.
- The current runtime port reproduces the intended V9 family direction, but the latest single-seed numeric result differs from the older v4-v9 note: direction improves much more and root height drops to `0.117m`. This is likely due to using the production-side extra-loss integration plus deterministic effective-candidate seeds rather than the earlier monkey-patch-only pass.

## Follow-Up

- Use the viewer command below for human visual inspection.
- Do not promote V9 as a fix: touchdown IK/FK still worsens in this latest run (`0.661772 -> 0.745050`) and root height is low.

## Git Refs

- Baseline Ref: working tree before V9 runtime port
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py](../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
