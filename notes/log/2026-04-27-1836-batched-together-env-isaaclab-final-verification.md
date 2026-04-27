# Batched Together Env IsaacLab Final Verification

## Purpose

Record the final reviewed verification pass for the native IsaacLab GPU `batched_together_planner` migration after subagent implementation and smoke fixes.

## Stage

`batched_together_planner` core, manager/factory/reward wiring, train/play attach path, and viewer backend path.

## Related Todo

[T100 batched together planner GPU migration](../todo/T100-batched-together-planner-gpu-migration.md)

## Command / Procedure

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q -s Go2Pvcnn/tests/test_batched_together_guardrails.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_manager.py Go2Pvcnn/tests/test_batched_together_parity.py Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m py_compile Go2Pvcnn/extension/__init__.py Go2Pvcnn/extension/trajectory_manager_factory.py Go2Pvcnn/extension/mdp/rewards_reference.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/scripts/train.py Go2Pvcnn/scripts/play.py Go2Pvcnn/extension/batched_together_planner/*.py Go2Pvcnn/tests/test_batched_together_*.py Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py
rg -n "import numpy|from numpy|np\.|\.cpu\(|\.item\(|\.numpy\(|\.tolist\(|nonzero\(|index_select\(|index_copy_|masked_select|torch\.cuda\.synchronize|torch\.allclose|torch\.equal" Go2Pvcnn/extension/batched_together_planner Go2Pvcnn/extension/trajectory_manager_factory.py Go2Pvcnn/extension/mdp/rewards_reference.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/scripts/train.py Go2Pvcnn/scripts/play.py
git diff --check -- Go2Pvcnn/extension Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/scripts/train.py Go2Pvcnn/scripts/play.py Go2Pvcnn/tests notes
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64: timeout -s INT -k 20s 60s /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --device cuda:0 --num_envs 1 --terrain task --planner-backend together --n-frames 35 --plan-dt 0.02 --warmup-steps 0 --scripted-command "0.20 0.00 0.00" --scripted-command-cycles 1
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

Subagent runtime smoke also ran `train.py` and `play.py` in `/home/lhy/anaconda3/envs/env_isaaclab` for both `together` and `legacy` with short, bounded settings.

## Input Conditions

- Candidate ref: working tree on top of `7cf6c11`.
- Python: `/home/lhy/anaconda3/envs/env_isaaclab/bin/python`.
- CUDA: available, 2x RTX 4090 in the conda environment.
- Isaac viewer: headless; `DISPLAY=None`; stdin non-TTY, so teleop disabled and scripted command used.

## Key Metrics / Evidence

- Full together test suite: `50 passed in 0.97s`.
- CUDA smoke printed:
  - `N=1024 elapsed_ms=24.324 cuda_allocated_mb=52.40`
  - `N=4096 elapsed_ms=1.903 cuda_allocated_mb=204.80`
- `py_compile`: pass.
- `git diff --check`: pass.
- Training-path guardrail tests are included in the `50 passed` suite and check the together planner manifest for forbidden CPU packages/sync and forbidden hot-path loops.
- Manual broad token scan over `train.py` / `play.py` found existing non-together camera/debug tokens:
  - `Go2Pvcnn/scripts/play.py` uses NumPy and `.detach().cpu().numpy()` for follow-camera playback.
  - `Go2Pvcnn/scripts/train.py` calls `torch.cuda.synchronize()` during startup CUDA diagnostics.
  These are outside `extension/batched_together_planner` and outside the scoped training-path guardrail, but they remain visible if scanning the entire script files naively.
- Viewer together smoke evidence:
  - `[Viewer] Attached together trajectory manager`
  - `[Viewer][Plan] backend=together cycle=0 cmd=(+0.20, +0.00, +0.00) ... standstill=False`
  - `[Viewer][Playback] path=render+scene_sync`
  - `frame=34/35`
- GPU process cleanup: final `nvidia-smi --query-compute-apps` returned no rows.
- Subagent Isaac smoke evidence:
  - `train.py --planner-backend together --max_iterations 0`: env creation, together manager attach, wrapper reset, runner creation, normal exit.
  - `train.py --planner-backend legacy --max_iterations 0`: legacy attach, wrapper reset, runner creation, normal exit after factory hook compatibility fix.
  - `play.py --planner-backend together --video_length 1`: env creation, manager attach, checkpoint load, one timestep.
  - `play.py --planner-backend legacy --video_length 1`: same rollback smoke.

## Result

Pass with scoped caveats.

The implemented together path passes the conda/CUDA pytest suite, static training-path guardrails, py_compile, diff whitespace checks, and short IsaacLab runtime smoke. Viewer `together` now actually routes scripted/teleop commands into `extension.batched_together_planner.planner.plan_segment`.

## Conclusion

T100 is implemented and smoke-verified for the intended short training/runtime/viewer paths. The remaining work is not basic wiring; it is broader raw semantic coverage and longer runtime performance validation.

## Follow-Up

- T103 remains open for complex terrain/support/CEM parity beyond flat P0 parity.
- T107 remains open for long training and large multi-env Isaac runtime throughput.
- Viewer still displays env0 only.
- Whole-script scans still see pre-existing `play.py` camera NumPy and `train.py` startup CUDA sync; the together training-path guardrail remains scoped to the new planner/factory/reward/env wiring.

## Git Refs

- Baseline Ref: `7cf6c11`
- Candidate Ref: working tree on top of `7cf6c11` at `2026-04-27 18:36 +0800`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner](../../Go2Pvcnn/extension/batched_together_planner)
  - [../../Go2Pvcnn/extension/trajectory_manager_factory.py](../../Go2Pvcnn/extension/trajectory_manager_factory.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py)
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
  - [../../Go2Pvcnn/tests/test_batched_together_runtime_path.py](../../Go2Pvcnn/tests/test_batched_together_runtime_path.py)
