# 2026-06-13 18:39 Play Terminal Keyboard Backend

## Purpose

按用户要求移除 `pynput` 路线：`play.py --keyboard-control` 不再依赖 X/DISPLAY，也不新增 backend 参数，而是从启动命令的终端并行读取键盘输入。

## Stage

Play entrypoint / livestream terminal control.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

RED tests:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py::test_play_cli_has_keyboard_control_and_terrain_selection_flags \
  Go2Pvcnn/tests/test_viewer_reset.py::test_keyboard_velocity_controller_uses_terminal_reader_thread_not_pynput \
  -q
```

Observed before implementation:

```text
2 failed
```

Focused GREEN:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py::test_play_cli_has_keyboard_control_and_terrain_selection_flags \
  Go2Pvcnn/tests/test_viewer_reset.py::test_keyboard_velocity_controller_uses_terminal_reader_thread_not_pynput \
  Go2Pvcnn/tests/test_viewer_reset.py::test_keyboard_velocity_controller_maps_pressed_keys_to_body_command \
  Go2Pvcnn/tests/test_viewer_reset.py::test_keyboard_velocity_controller_speed_step_scales_and_clamps \
  Go2Pvcnn/tests/test_viewer_reset.py::test_apply_keyboard_command_overwrites_base_velocity_tensor \
  -q
```

Full static:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q
python -m py_compile Go2Pvcnn/scripts/play.py
```

Real smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/play.py \
  --headless --device cuda:0 --num_envs 1 --max-steps 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --run_dir 2026-06-12_19-05-27 --checkpoint model_28900.pt \
  --terrain-row 3 --terrain-col 0 --keyboard-control
```

## Input Conditions

- Current automated tool runner stdin is not a TTY.
- User's intended livestream usage runs from an SSH terminal, where stdin should be a TTY.
- No new CLI backend parameter is allowed.

## Key Evidence

- Focused GREEN:

```text
5 passed
```

- Full viewer reset:

```text
32 passed
```

- `python -m py_compile Go2Pvcnn/scripts/play.py` exits `0`.
- Real smoke exits `0`.
- Real smoke prints:

```text
[play.py] Initial terrain env0: row=3, col=0
[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
[WARN][play.py] --keyboard-control disabled: failed to start terminal keyboard reader: stdin is not a TTY
```

## Result

Pass. The old `pynput` import path is removed from `play.py`; `--keyboard-control` now starts a terminal reader thread when stdin is a TTY.

## Conclusion

For livestream usage, the browser remains the visual output and the SSH terminal is now the keyboard input source. Non-TTY launchers warn and continue without keyboard control.

## Follow-Up

Run the livestream command directly from an interactive SSH terminal to verify live key control by hand.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
  - [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)
