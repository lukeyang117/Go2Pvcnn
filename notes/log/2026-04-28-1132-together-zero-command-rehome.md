# Together Zero-Command Rehome Recovery

## Purpose

Fix the viewer-reported stop-command issue where the together backend could leave Go2 crouched or twisted after velocity command `0`, because the zero-command hold path did not actively recover root posture/height.

## Stage

`Go2Pvcnn/extension/batched_together_planner` planner core, T100/T110.

## Related Todo

[T100 batched together planner GPU migration](../todo/T100-batched-together-planner-gpu-migration.md#t110-zero-command-rehome-upright-recovery)

## Command / Procedure

TDD red test from worker:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_batched_together_core.py::test_zero_command_root_frame_rehome_moves_terminal_feet_toward_nominal
```

Expected failure before implementation:

```text
terminal root_pos[:, -1, 2] stayed frozen instead of recovering to cfg.hip_height;
3/3 mismatched, greatest absolute diff 0.1200000048
```

Green verification:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_parity.py Go2Pvcnn/tests/test_batched_together_guardrails.py
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_viz_playback.py
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_parity.py Go2Pvcnn/tests/test_batched_together_guardrails.py Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_viz_playback.py
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m py_compile Go2Pvcnn/extension/batched_together_planner/config.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/tests/test_batched_together_guardrails.py
```

Diagnostic script:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python - <<'PY'
# Builds a bad-pose zero-command together plan and prints initial/terminal root state.
PY
```

Headless viewer smoke:

```bash
timeout -s INT -k 20s 120s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64:${LD_LIBRARY_PATH:-} PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 0 --device cuda:0 --num_envs 1 --warmup-steps 0 --scripted-command "0.40 0.00 0.00" --scripted-command-cycles 1 --planner-backend together --n-frames 35 --plan-dt 0.02
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

## Input Conditions

- Python: `/home/lhy/anaconda3/envs/env_isaaclab/bin/python`
- Planner backend: `together`
- Zero-command test state: nonzero roll/pitch/yaw, perturbed feet, and root height below/above nominal.
- Viewer smoke: one scripted forward cycle, then zero command.

## Key Metrics

- T110 core/parity/guardrail: `26 passed`.
- Runtime + viz playback regression: `29 passed`.
- Combined together core/parity/guardrail/runtime/viz subset: `55 passed`.
- `py_compile`: pass.
- CUDA diagnostic:
  - initial rpy: `[0.22, -0.16, 0.70]`
  - terminal rpy: `[0.0, 0.0, 0.70]`
  - initial root z: `0.18`
  - terminal root z: `0.30`
  - contact all: `True`
  - touchdown any: `False`
- Headless viewer smoke reached real Isaac startup, together manager attachment, forward planning/playback, zero-command replanning, and repeated playback near `plan.z=+0.30`, `actual.z=+0.30`, `plan_rpy=(+0.000, +0.000, +0.000)`.
- Viewer smoke exited by timeout as expected for the long-running viewer loop.
- `nvidia-smi --query-compute-apps`: no residual compute process.

## Result

Pass with scoped caveat.

The together zero-command hold path now keeps root `xy` and yaw, moves root `z` toward terrain support plus `hip_height`, moves roll/pitch toward the support plane, and moves feet toward root-frame nominal slots with support-height z. Mixed zero/moving rows remain selected by tensor masks.

## Parallelism / Guardrail Notes

- The implementation uses batched tensor operations and `torch.where`; no dynamic env sub-batch is introduced.
- `torch.linalg.svd` was deliberately removed from the support-plane target and replaced by a vectorized four-foot midpoint cross-product normal.
- The static guardrail now forbids `torch.linalg.svd` and `torch.svd` in the together training path.
- The scanned training path remains free of NumPy/CPU numeric imports, Python loops/comprehensions, CPU sync calls, and forbidden dynamic indexing calls.

## Conclusion

The viewer symptom was not only a viewer issue: together planner core had an incomplete zero-command rehome. The fix aligns the P0 flat stop behavior with raw-style upright recovery while preserving the fixed-shape full-batch training contract.

## Follow-Up

- Complex terrain raw parity remains under T103; this log only verifies flat/P0 support and viewer stop smoke.
- Manual interactive viewer confirmation is still useful after the earlier root-z ratchet and this zero-command recovery fix.

## Git Refs

- Baseline Ref: `7cf6c11`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/config.py](../../Go2Pvcnn/extension/batched_together_planner/config.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/parameterization.py](../../Go2Pvcnn/extension/batched_together_planner/parameterization.py)
  - [../../Go2Pvcnn/tests/test_batched_together_core.py](../../Go2Pvcnn/tests/test_batched_together_core.py)
  - [../../Go2Pvcnn/tests/test_batched_together_parity.py](../../Go2Pvcnn/tests/test_batched_together_parity.py)
  - [../../Go2Pvcnn/tests/test_batched_together_guardrails.py](../../Go2Pvcnn/tests/test_batched_together_guardrails.py)
