# Viewer Together Root-Z Ratchet Fix

## Purpose

Analyze and fix the viewer issue reported by the user: the Go2 visually walks for a while and then lifts into the air when launching the together viewer.

## Stage

`extension/viz` viewer playback and together planner handoff, T100/T109.

## Related Todo

[T100 batched together planner GPU migration](../todo/T100-batched-together-planner-gpu-migration.md#t109-viewer-together-root-z-ratchet)

## Command / Procedure

Root-cause pass:

- Read planner runtime notes and current T100 logs.
- Compared viewer loop handoff against together planner root-z trajectory.
- Identified that `_together_state_from_reference_result()` used the segment's last root z directly as the next segment initial state.

Red test:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_viz_playback.py::TestKinematicPlaybackLogic::test_together_viewer_handoff_does_not_accumulate_root_height_on_flat_walk
```

Initial expected failure:

```text
max_terminal_z=0.3964, initial_root_z=0.3000, threshold=0.3400
```

Green/regression commands:

```bash
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_viz_playback.py
PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python -m pytest -q --tb=short Go2Pvcnn/tests/test_batched_together_parity.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_runtime_path.py Go2Pvcnn/tests/test_batched_together_guardrails.py
/home/lhy/anaconda3/envs/env_isaaclab/bin/python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_viz_playback.py
timeout -s INT -k 20s 120s env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lhy/cuda-12.2/lib64:${LD_LIBRARY_PATH:-} PYTHONPATH=/home/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/home/lhy/testPvcnnWithIsaacsim /home/lhy/anaconda3/envs/env_isaaclab/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 0 --device cuda:0 --num_envs 1 --warmup-steps 0 --scripted-command "0.40 0.00 0.00" --scripted-command-cycles 4 --planner-backend together --n-frames 35 --plan-dt 0.02
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

## Input Conditions

- Python: `/home/lhy/anaconda3/envs/env_isaaclab/bin/python`
- Viewer backend: `together`
- Scripted viewer command: `(0.40, 0.00, 0.00)`, 4 walking cycles, then zero command.
- Candidate ref: working tree on top of `7cf6c11`.

## Key Metrics

- New regression test failed before the fix with terminal z reaching `0.3964m` from `0.3000m` after 4 viewer-style replans.
- `test_viz_playback.py`: `20 passed`.
- Together parity/core/runtime/guardrail set: `35 passed`.
- `py_compile`: pass.
- Headless viewer reached real Isaac startup, attached together manager, planned and played together trajectories; it was terminated by `timeout` because the viewer loop is long-running by design.
- Viewer smoke evidence after walking cycles showed subsequent standstill frames at about `plan.z=+0.30`, `actual.z=+0.30`, not a continuing z climb.
- `nvidia-smi --query-compute-apps`: no residual compute process.

## Result

Pass with scoped caveat.

The implemented fix keeps raw/together single-segment parity intact and changes only the together viewer handoff state: the next segment's root z is reconstructed from current support-foot height plus the segment's initial support clearance. This prevents viewer-only segment z-bias from being treated as permanent base height.

## Conclusion

The user-visible lift-off was rooted in viewer segment chaining, not in viewer startup or direct pose writeback. The fix is isolated to `extension/viz/go2_foostep_planner.py`; training-path guardrails remain unchanged.

## Follow-Up

- User should rerun the interactive viewer to confirm the visual no longer lifts off under manual teleop.
- Training long-run behavior remains governed by T107/T103 and was not retested here.

## Git Refs

- Baseline Ref: `7cf6c11`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viz_playback.py](../../Go2Pvcnn/tests/test_viz_playback.py)
