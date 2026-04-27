# Batched Together Continued Testing

## Purpose

Continue testing after the initial `env_isaaclab` smoke by covering short PPO training, real IsaacLab cadence/full-N behavior, and broader regression tests around the legacy/default-together contract.

## Stage

`batched_together_planner` training runtime, manager cadence, viewer/play/train helper regression.

## Related Todo

[T100 batched together planner GPU migration](../todo/T100-batched-together-planner-gpu-migration.md)

## Command / Procedure

Subagents executed:

```bash
timeout -s INT -k 60s 900s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64:$LD_LIBRARY_PATH PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /usr/bin/time -f WALL_SECONDS=%e /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/scripts/train.py --headless --device cuda:0 --num_envs 32 --max_iterations 1 --experiment teacher_elevation_trajectory --planner-backend together
timeout -s INT -k 60s 900s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64:$LD_LIBRARY_PATH PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /usr/bin/time -f WALL_SECONDS=%e /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/scripts/train.py --headless --device cuda:0 --num_envs 128 --max_iterations 1 --experiment teacher_elevation_trajectory --planner-backend together
timeout -s INT -k 60s 900s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64:$LD_LIBRARY_PATH PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /usr/bin/time -f WALL_SECONDS=%e /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/scripts/train.py --headless --device cuda:0 --num_envs 16 --max_iterations 1 --experiment teacher_elevation_trajectory --planner-backend legacy
timeout -s INT -k 30s 300s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64:${LD_LIBRARY_PATH:-} PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python - --headless --device cuda:0 --test-num-envs 4
timeout 300 /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_batched_reference_integration.py -q --tb=short
timeout 900 /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q --tb=short
timeout 900 /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest Go2Pvcnn/tests/test_batched_planner_stage_diagnostics.py -q --tb=short
```

Main-agent verification:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_batched_planner_runtime_path.py Go2Pvcnn/tests/test_batched_planner_instrumentation.py Go2Pvcnn/tests/test_batched_reference_integration.py Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_batched_together_guardrails.py
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q -s Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_manager.py Go2Pvcnn/tests/test_batched_together_parity.py Go2Pvcnn/tests/benchmarks/bench_batched_together_planner.py
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m py_compile Go2Pvcnn/tests/test_batched_planner_runtime_path.py Go2Pvcnn/tests/test_batched_planner_instrumentation.py Go2Pvcnn/tests/test_batched_reference_integration.py Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_batched_together_guardrails.py
git diff --check -- Go2Pvcnn/tests/test_batched_planner_runtime_path.py Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_batched_together_guardrails.py Go2Pvcnn/extension Go2Pvcnn/scripts Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py notes
timeout -s INT -k 60s 900s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64:$LD_LIBRARY_PATH PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /usr/bin/time -f WALL_SECONDS=%e /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/scripts/train.py --headless --device cuda:0 --num_envs 32 --max_iterations 1 --experiment teacher_elevation_trajectory --planner-backend together
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

## Input Conditions

- Candidate ref: working tree on top of `7cf6c11`.
- Python: `/home/lhy/anaconda3/envs/env_isaaclab/bin/python`.
- CUDA device: `cuda:0`.
- Tests ran headless.

## Key Metrics / Evidence

- Subagent train smoke:
  - together `32` env, `max_iterations=1`: exit `0`, `WALL_SECONDS=10.59`, `Total steps: 1280`, `Steps per second: 781`, `Training Complete!`
  - together `128` env, `max_iterations=1`: exit `0`, `WALL_SECONDS=10.63`, `Total steps: 5120`, `Steps per second: 2683`, `Training Complete!`
  - legacy `16` env, `max_iterations=1`: exit `0`, `WALL_SECONDS=10.60`, `Total steps: 640`, `Steps per second: 335`, `Training Complete!`
- Main-agent train recheck:
  - together `32` env, `max_iterations=1`: exit `0`, `WALL_SECONDS=10.19`, `Total steps: 1280`, `Steps per second: 816`, `Training Complete!`
- Real env cadence/full-N:
  - `num_envs=4`, real Isaac env and `TogetherTrajectoryManager`.
  - Planner call batch sizes stayed `[4, 4, 4, 4]` for reset, one-env `mark_command_changed`, 35-frame interval, and real command term `_resample_command([2])` hook.
  - Same host step refresh did not duplicate planning.
  - Interval matched `35 * 0.02 = 0.7000000000000001s`.
- Broader regression:
  - `test_batched_reference_integration.py`: `8 passed`.
  - `test_viewer_runtime_diagnostics.py`: `10 passed`.
  - `test_batched_planner_stage_diagnostics.py`: `8 passed`.
  - Initial old runtime test found `5` contract-drift failures; `Go2Pvcnn/tests/test_batched_planner_runtime_path.py` was updated to the current factory/default-together/viewer-task contract.
- Main-agent pytest:
  - runtime/regression/guardrail set: `59 passed in 0.46s`.
  - together core/manager/parity/benchmark set: `36 passed in 1.00s`.
  - CUDA benchmark printed `N=1024 elapsed_ms=27.867 cuda_allocated_mb=52.40` and `N=4096 elapsed_ms=2.004 cuda_allocated_mb=204.80`.
- `py_compile`: pass.
- `git diff --check`: pass.
- Final `nvidia-smi --query-compute-apps`: no rows.

New train output directories were created:

- `logs/rsl_rl/teacher_elevation_trajectory/2026-04-27_19-06-14`
- `logs/rsl_rl/teacher_elevation_trajectory/2026-04-27_19-06-46`
- `logs/rsl_rl/teacher_elevation_trajectory/2026-04-27_19-07-15`
- `logs/rsl_rl/teacher_elevation_trajectory/2026-04-27_19-13-54`

## Result

Pass with follow-up caveats.

The together path now has short PPO training evidence at `32` and `128` envs, real env cadence/full-N evidence, broader legacy/reference/viewer regression evidence, and updated tests for the new backend factory contract.

## Conclusion

This round moved T107 from startup smoke to short one-iteration training and real cadence validation. It did not reveal production-code regressions. One test-only regression was fixed because old tests still expected legacy-only attachment and viewer defaults.

## Follow-Up

- Still not covered: multi-iteration training, env counts beyond `128`, multi-device training, long-run memory/throughput drift, and complex terrain/support/CEM semantic parity.
- Isaac emitted usual headless/CUDA_VISIBLE_DEVICES warnings, but runs completed and no residual compute process remained.

## Git Refs

- Baseline Ref: `7cf6c11`
- Candidate Ref: working tree on top of `7cf6c11` at `2026-04-27 19:14 +0800`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner](../../Go2Pvcnn/extension/batched_together_planner)
  - [../../Go2Pvcnn/extension/trajectory_manager_factory.py](../../Go2Pvcnn/extension/trajectory_manager_factory.py)
  - [../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py](../../Go2Pvcnn/tests/test_batched_planner_runtime_path.py)
  - [../../Go2Pvcnn/tests/test_batched_together_runtime_path.py](../../Go2Pvcnn/tests/test_batched_together_runtime_path.py)
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
