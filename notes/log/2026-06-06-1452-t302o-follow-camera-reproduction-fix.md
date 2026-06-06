# T302o Follow Camera Reproduction Fix

## Purpose

Reproduce and fix the user-reported issue that `mpc_policy_eval.py --livestream 2 --num-envs 1` still did not lock/follow the robot view.

## Stage

MPC semantic policy evaluation / livestream follow camera.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Baseline Ref

- Working tree after the earlier livestream marker/follow-camera smoke in [2026-06-06-1413-t302o-livestream-marker-follow-camera.md](2026-06-06-1413-t302o-livestream-marker-follow-camera.md).

## Candidate Ref

- Working tree after preserving `livestream_enabled` before `AppLauncher(args)` and decoupling follow-camera updates from the marker branch in [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py).

## Current Work Ref

- Branch: `costmap-teacher-ablation`

## Key Files

- [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
- [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)

## Command / Procedure

Local verification:

```bash
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

Reproduction run before the fix:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --livestream 2 \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 10 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "1.0 0.0 0.0" \
  --debug-follow-camera \
  --output-dir logs/mpc_policy_eval/debug_follow_camera
```

Verification run after the fix:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --livestream 2 \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 10 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "1.0 0.0 0.0" \
  --debug-follow-camera \
  --output-dir logs/mpc_policy_eval/debug_follow_camera_fixed_branch
```

## Input Conditions

- GPU selection: `CUDA_VISIBLE_DEVICES=0`
- Python: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Livestream mode: `--livestream 2`
- Env count: `--num-envs 1`
- Steps: `--max-steps 10`

## Key Metrics

- Local pycompile: exit `0`.
- Local static + metric regression: `18 passed in 1.68s`.
- Reproduction output: `logs/mpc_policy_eval/debug_follow_camera/2026-06-06_14-49-15-557087`
  - exit `0`
  - `Streaming server started`
  - `metrics.jsonl`: `10` lines
  - `follow_camera_debug.jsonl`: `0` lines
- Fixed output: `logs/mpc_policy_eval/debug_follow_camera_fixed_branch/2026-06-06_14-51-25-796244`
  - exit `0`
  - `Streaming server started`
  - `metrics.jsonl`: `10` lines
  - `follow_camera_debug.jsonl`: `10` lines
  - `active_viewport_camera_path`: `/OmniverseKit_Persp`
  - `active_camera_world_position` matches `requested_camera_position` on sampled rows
  - `reference_valid_ratio`: `1.0`
  - `tracking_valid_step_count`: `10`

## Result

Pass for the reproduced failure mode.

## Conclusion

The previous render-timing hypothesis was incomplete. The reproduced failure showed that the rollout completed while `follow_camera_debug.jsonl` stayed empty, so the follow-camera branch was not being executed at all. The cause was relying on `args.livestream` after `AppLauncher(args)`; AppLauncher mutates/consumes launcher args, making later livestream checks unreliable. Because marker creation and follow-camera were both gated by those later checks, the camera update path was skipped.

The fix snapshots `livestream_enabled` before constructing `AppLauncher`, uses that stable value for `render_mode`, marker creation, and follow-camera updates, and moves follow-camera out of the marker branch so env-one camera follow does not depend on marker construction.

## Follow-Up

- Browser-side visual confirmation still depends on the user's client view, but runtime evidence now shows the active viewport camera is updated every step.
- Terrain row/col selector semantics remain a separate T302o follow-up.

