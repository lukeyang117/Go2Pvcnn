# Joint MPC RTI Viewer CUDA Graph OOM Fallback

- Purpose: fix the real livestream viewer crash where the second `joint_mpc_rti` refresh failed at CUDA Graph `capture_end()` with `CUDA error: out of memory`.
- Stage: `joint_mpc_rti` runtime manager / viewer refresh path.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: current `joint_mpc_kinematic` worktree before this fix.
- Candidate Ref: current working tree.
- Key Files:
  - `Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py`
  - `Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py`

## Root Cause

The viewer first refresh could run eagerly and publish a result. The next refresh tried to create `JointMpcCudaGraphRunner`; under the current Isaac/RTX/WebRTC memory pressure, CUDA Graph capture failed at `capture_end()` with OOM. The runtime manager did not catch recoverable graph-capture failures, so the whole viewer exited instead of continuing with eager RTI.

## Change

- Added recoverable CUDA Graph error detection for:
  - `CUDA error: out of memory`
  - `CUDA out of memory`
  - `operation not permitted when stream is capturing`
- If graph construction/rebuild/replay fails with one of those errors:
  - clear the graph runner,
  - disable CUDA Graph for this manager instance,
  - call `torch.cuda.empty_cache()`,
  - print a warning,
  - rerun the same refresh with eager `planner_step`.

No RTI objective, constraints, nominal, terrain, or line-search formulas were changed.

## Verification

RED before fix:

```text
test_cuda_graph_oom_falls_back_to_eager_for_viewer_runtime
RuntimeError: CUDA error: out of memory
```

GREEN after fix:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py::test_cuda_graph_oom_falls_back_to_eager_for_viewer_runtime -q
1 passed in 8.01s
```

Runtime regression:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py -q
12 passed in 485.01s
```

Static checks:

```text
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py
exit 0

git diff --check -- Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py Go2Pvcnn/tests/joint_mpc_rti/test_rolling_runtime.py
exit 0
```

Viewer smoke with the user's command shape:

```text
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 timeout 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend joint_mpc_rti --n-frames 30 --plan-dt 0.02 --terrain-row 0 --terrain-col 0 --warmup-steps 0
```

The viewer reached RTI playback, hit the same graph OOM, printed the fallback warning, and continued with eager RTI instead of raising the traceback:

```text
[JointMpcRTI][WARN] CUDA Graph failed during runtime; falling back to eager RTI for this manager. error=CUDA error: out of memory
```

## Boundary

After fallback, the smoke later reported `kkt=(nan,nan)` / `clearance=-inf` from phase 6 onward. That is a separate planner numerical/validity issue and is not fixed by this CUDA Graph OOM fallback. The WebRTC stream also printed `nvstPushStreamData timeout`, likely because the smoke ran under timeout/tee without an attached browser client and on a busy GPU.

