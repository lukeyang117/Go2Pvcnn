# 2026-06-13 17:56 Play Pynput Install Headless Smoke

## Purpose

安装并测试 `env_isaacsim` 里的 `pynput`，确认 `play.py --keyboard-control` 在当前 headless 终端下的实际行为。

## Stage

Play entrypoint / keyboard command backend.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

Dependency install:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pip install pynput
```

Dependency checks:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pip show pynput

/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python - <<'PY'
import importlib
import importlib.metadata
print('pynput version:', importlib.metadata.version('pynput'))
try:
    importlib.import_module('pynput.keyboard')
    print('pynput.keyboard import: OK')
except Exception as exc:
    print('pynput.keyboard import: FAIL')
    print(type(exc).__name__ + ': ' + str(exc))
PY
```

Static verification:

```bash
pytest Go2Pvcnn/tests/test_viewer_reset.py -q
python -m py_compile Go2Pvcnn/scripts/play.py
```

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/play.py \
  --headless --device cuda:0 --num_envs 1 --max-steps 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --run_dir 2026-06-12_19-05-27 --checkpoint model_28900.pt \
  --terrain-row 3 --terrain-col 0 --keyboard-control
```

## Input Conditions

- Python env: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`
- Experiment: `teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance`
- Checkpoint: `2026-06-12_19-05-27/model_28900.pt`
- Runtime mode: `--headless`, `--num_envs 1`, `--max-steps 1`, `--keyboard-control`
- Current shell has no valid X `DISPLAY`.

## Key Evidence

- `pynput` install succeeded.
- `pip show pynput` reports:

```text
Name: pynput
Version: 1.8.2
Location: /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/lib/python3.10/site-packages
Requires: evdev, python-xlib, six
```

- `pynput.keyboard` import fails in the current headless shell:

```text
ImportError: this platform is not supported: ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
```

- Focused static verification:

```text
31 passed in 1.42s
```

- `python -m py_compile Go2Pvcnn/scripts/play.py` exits `0`.
- Real smoke exits `0`.
- Real smoke prints:

```text
[play.py] Initial terrain env0: row=3, col=0
[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
[WARN][play.py] --keyboard-control disabled: failed to import pynput.keyboard: this platform is not supported: ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
```

## Result

Pass for dependency installation, static tests, compile, and guarded runtime smoke.

Interactive `pynput` keyboard capture is not available in the current headless shell because there is no X server / valid `DISPLAY`. The `play.py` guard handles this case by warning and continuing instead of crashing.

## Conclusion

The dependency problem is solved. The remaining limitation is the keyboard backend: `pynput` needs access to an X keyboard session, so browser/WebRTC or pure headless SSH input will not automatically reach this backend.

## Follow-Up

For headless SSH control, add a terminal raw-input backend or another input channel instead of relying only on `pynput`.

## Git Refs

- Baseline Ref: `23182ce`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/tests/test_viewer_reset.py](../../Go2Pvcnn/tests/test_viewer_reset.py)
