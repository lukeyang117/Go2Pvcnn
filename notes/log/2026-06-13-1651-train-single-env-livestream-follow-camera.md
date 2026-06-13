# 2026-06-13 16:51 Train Single-Env Livestream Follow Camera

## Purpose

让 `Go2Pvcnn/scripts/train.py` 在 `--num_envs 1 --livestream 1/2` 训练可视化时跟随 env0 机器人，便于观察 flat-small avoidance 场景里的小语义物体和机器人交互。

## Stage

Training entrypoint / WebRTC visualization.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

Static RED/GREEN:

```bash
pytest Go2Pvcnn/tests/test_train_script_static.py -q
```

Syntax check:

```bash
python -m py_compile Go2Pvcnn/scripts/train.py
```

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless --livestream 2 --device cuda:0 --num_envs 1 --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07 \
  --load_checkpoint model_19999.pt
```

## Input Conditions

- Checkpoint: `model_19999.pt`
- Experiment: `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance`
- `num_envs=1`
- `livestream=2`
- Device mapping: `CUDA_VISIBLE_DEVICES=2`, script device `cuda:0`

## Key Evidence

- RED before implementation: static test failed because `_SingleEnvLivestreamFollowCamera` was absent.
- First real smoke exposed that `AppLauncher` can mutate `args_cli.livestream`; follow-camera install did not print.
- Added `requested_livestream` before `_launch_app(args_cli)` and pass that preserved value into camera installation.
- GREEN static result: `2 passed`.
- Pycompile exit: `0`.
- Real smoke exit: `0`.
- Real smoke printed:

```text
[train.py] Single-env livestream follow camera enabled (interval=4 env steps, env0 root).
```

## Result

Pass. Training now installs a follow camera only for rank 0, one-env livestream runs, and updates the camera from env0 robot root after environment steps.

## Conclusion

The user command with `--num_envs 1 --livestream 2` should now keep the WebRTC view locked near the robot during training. Multi-env and non-livestream training paths are left unchanged.

## Follow-Up

If tiny semantic objects are still hard to see, the next change should be visual styling/marker support for semantic obstacles, not broader camera changes.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - [../../Go2Pvcnn/tests/test_train_script_static.py](../../Go2Pvcnn/tests/test_train_script_static.py)
