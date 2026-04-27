# Viewer Together Backend Smoke

## Purpose

Verify that `Go2Pvcnn/extension/viz/go2_foostep_planner.py --planner-backend together` now calls the native `extension.batched_together_planner` planning path, while `--planner-backend legacy` preserves the old batched planner path.

## Stage

`extension/viz` viewer runtime, T100 batched together planner migration.

## Related Todo

[T100 batched together planner GPU migration](../todo/T100-batched-together-planner-gpu-migration.md)

## Command / Procedure

```bash
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batched_together_runtime_path.py
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_batched_together_runtime_path.py -q
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_batched_together_guardrails.py -q
timeout 90s /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 0 --num_envs 1 --warmup-steps 0 --scripted-command "0.20 0.00 0.00" --scripted-command-cycles 2 --planner-backend together
timeout 90s /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 0 --num_envs 1 --warmup-steps 0 --scripted-command "0.20 0.00 0.00" --scripted-command-cycles 2 --planner-backend legacy
```

The two viewer smoke commands were filtered to evidence lines for the final pass.

## Input Conditions

- Candidate ref: working tree on top of `7cf6c11`.
- Device: Isaac Lab `cuda:0`.
- Viewer command: scripted body-frame command `(0.20, 0.00, 0.00)`.
- Scope: viewer file plus viewer-focused runtime tests only.

## Key Metrics / Evidence

- Runtime path tests: `9 passed`.
- Guardrail tests: `5 passed`.
- Follow-up review fixed a together viewer config drift: `_build_together_planner_cfg()` now keeps raw defaults for `duty_factor`, `support_search_radius`, and `support_search_step` even when legacy env cfg fields are present; explicit together fields still override.
- Together smoke evidence:
  - `[Viewer] Attached together trajectory manager`
  - `[Viewer][Plan] backend=together cycle=0 cmd=(+0.20, +0.00, +0.00) ... standstill=False`
  - `[Viewer][Playback] path=render+scene_sync`
  - `frame=34/35`
- Legacy smoke evidence:
  - `[Viewer] Attached legacy trajectory manager`
  - `[Viewer][Plan] backend=legacy cycle=0 cmd=(+0.20, +0.00, +0.00) ... standstill=False`
  - `[Viewer][Playback] path=render+scene_sync`
  - `frame=24/25`

## Result

Pass. The viewer together backend now routes the scripted/teleop command into `plan_segment`; legacy remains on `batched_generate_trajectory`.

## Conclusion

The viewer backend selector is aligned with the T100 together migration without expanding training-path guardrail scanning into `extension/viz`.

## Follow-Up

Broader viewer visual quality and multi-env display behavior remain outside this smoke; viewer still displays env0.

## Git Refs

- Baseline Ref: `7cf6c11`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_batched_together_runtime_path.py](../../Go2Pvcnn/tests/test_batched_together_runtime_path.py)
