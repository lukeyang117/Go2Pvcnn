# 2026-06-13 17:35 Play Keyboard Control Terrain Selection

## Purpose

给 `Go2Pvcnn/scripts/play.py` 增加 flat-small PLAY 可视化键盘控制：按住 `W/S/A/D/Q/E` 生成速度命令，`+/-` 调速度，并支持用 `--terrain-row` / `--terrain-col` 把 env0 初始化到指定子地形。

## Stage

Play entrypoint / flat-small semantic visualization.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

Focused tests:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_semantic_train_play_parsers_and_gym_registration_are_isolated \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_entrypoints_are_registered \
  -q
```

Syntax check:

```bash
python -m py_compile \
  Go2Pvcnn/scripts/play.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/play.py \
  --headless --device cuda:0 --num_envs 1 --max-steps 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --run_dir 2026-06-12_19-05-27 --checkpoint model_28900.pt \
  --terrain-row 3 --terrain-col 0
```

Dependency probe:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
try:
    from pynput import keyboard
    print('pynput.keyboard import: OK')
except Exception as exc:
    print(f'pynput.keyboard import: FAIL: {exc}')
PY
```

## Input Conditions

- Experiment: `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance`
- PLAY task id: `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-Play-v0`
- Checkpoint: `2026-06-12_19-05-27/model_28900.pt`
- Smoke terrain selection: `row=3`, `col=0`

## Key Evidence

- Initial RED: `5 failed, 24 passed` for missing keyboard helper/CLI/terrain selector.
- Additional RED found real PLAY reset issue: flat-small PLAY disabled semantic contact sensors but left `terrain_levels` curriculum active, causing reset to look up missing `semantic_contact_small`.
- GREEN focused tests: `34 passed`.
- Pycompile exit: `0`.
- Real smoke exit: `0`.
- Real smoke printed:

```text
[play.py] Initial terrain env0: row=3, col=0
[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
```

- Initial dependency probe reported `pynput.keyboard import: FAIL: No module named 'pynput'`.
- Follow-up install/testing is recorded in [2026-06-13-1756-play-pynput-install-headless-smoke.md](2026-06-13-1756-play-pynput-install-headless-smoke.md): `pynput 1.8.2` is now installed in `env_isaacsim`, but `pynput.keyboard` cannot initialize in the current headless shell because there is no X server / valid `DISPLAY`.

## Result

Pass for code wiring and flat-small PLAY runtime smoke. Keyboard control code is present and guarded. After installing `pynput`, interactive keyboard input still requires an X/DISPLAY keyboard session.

## Conclusion

`play.py` now supports hold-to-move keyboard command injection and deterministic env0 terrain row/col selection. The command is overwritten before policy observation and before `env.step`, so the policy sees the keyboard command rather than the random command manager sample.

## Follow-Up

Use a Python session where `pynput` can access an X/DISPLAY keyboard backend, or add a terminal/raw-input backend for headless SSH control.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
