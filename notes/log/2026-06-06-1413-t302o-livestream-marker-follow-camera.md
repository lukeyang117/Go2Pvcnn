# T302o Livestream Marker Follow Camera Fix

## Purpose

Fix and verify `mpc_policy_eval.py` livestream behavior after the user reported that MPC planned foot trajectories were not visible and env-count-one viewing should follow the robot.

## Stage

MPC semantic policy evaluation / livestream visualization.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Baseline Ref

- `9bd1e9f` after final T302o verification notes.

## Candidate Ref

- Working tree after adding full foot-trajectory markers and follow camera to [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py).

## Current Work Ref

- Branch: `costmap-teacher-ablation`

## Key Files

- [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
- [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)
- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)

## Command / Procedure

RED:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py::test_mpc_policy_eval_livestream_draws_full_foot_trajectories_and_follows_robot -q
```

Expected failures:

- first RED: script did not define `build_mpc_foot_trajectory_markers`;
- second RED after user still saw no camera follow: script did not call `base.sim.render()` after `set_camera_view()`.

GREEN:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py::test_mpc_policy_eval_livestream_draws_full_foot_trajectories_and_follows_robot -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

Real livestream smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --livestream 2 \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 2 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "1.0 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/visual_tracking_marker_follow_smoke
```

Camera render smoke after adding explicit render pump:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --livestream 2 \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 2 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "1.0 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/visual_tracking_camera_render_smoke
```

## Input Conditions

- GPU selection: `CUDA_VISIBLE_DEVICES=0`
- Python: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Livestream mode: `--livestream 2`
- Env count: `--num-envs 1`
- Existing unrelated dirty files preserved:
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - deleted legacy `.sh` files under `Go2Pvcnn/scripts/`
  - local NvStreamer `.etli` files

## Key Metrics

- RED: `1 failed`; missing full foot-trajectory marker helper.
- Camera RED: `1 failed`; missing `base.sim.render()` after camera update.
- Targeted GREEN: `1 passed in 0.01s`.
- Pycompile: exit `0`.
- Static + metric regression: `16 passed in 1.66s`.
- Livestream smoke: exit `0`; output `logs/mpc_policy_eval/visual_tracking_marker_follow_smoke/2026-06-06_14-13-07-351144`; `Streaming server started`; `total_steps=2`; `reference_valid_ratio=1.0`; tracking mean `0.014400195330381393`.
- Camera render smoke: exit `0`; output `logs/mpc_policy_eval/visual_tracking_camera_render_smoke/2026-06-06_14-30-05-152815`; `Streaming server started`; `total_steps=2`; `reference_valid_ratio=1.0`; tracking mean `0.017705533653497696`.

## Result

Pass for startup/runtime path.

## Conclusion

Root cause for missing trajectory was that `mpc_policy_eval.py` visualized only the current reference frame as four foot points. Viewer-style visibility requires plotting the full horizon per leg. The script now reads `_trajectory_reference_cache.foot_pos_w` as `[env, frame, leg, xyz]`, creates four colored marker groups under `/Visuals/T302oMpcPolicyEval/foot_traj_<leg>`, and visualizes `reference[0, :, leg_idx]` for each leg.

Root cause for camera not visibly following was render timing: `set_camera_view()` was called after `wrapped_env.step()`, but the script did not pump a render frame after the camera update. Viewer paths call `base_env.sim.render()` after visual/camera updates. The eval script now calls `base.sim.render()` immediately after `base.sim.set_camera_view(...)` in `update_follow_camera()`.

## Follow-Up

- Browser-side visual appearance was not inspected by this agent; the smoke verifies the marker and camera code path does not crash.
- Terrain row/col selector semantics remain a separate T302o follow-up.
