# T300d MPC Viewer Entrypoint + Replan Autograd Fix

- timestamp: 2026-05-11 15:05 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass

## Purpose

Fix the user's exact `env_isaacsim` viewer command for MPC backend:

```bash
python Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  --headless \
  --livestream 2 \
  --webrtc-public-ip 172.31.179.75 \
  --device cuda:2 \
  --num_envs 1 \
  --terrain task \
  --planner-backend mpc
```

## Stage

MPC viewer runtime path (`go2_foostep_planner.py`) from process bootstrap -> planner replan loop.

## Triggered Failures And Root Causes

1. `ModuleNotFoundError: No module named 'extension.batched_together_planner'`  
   Root cause: module-level `from extension...` import executed before `GO2PVCNN_ROOT` was inserted into `sys.path` when running the script directly.

2. `RuntimeError: Can't call numpy() on Tensor that requires grad` in `_update_camera`  
   Root cause: camera tensors were converted to NumPy without `detach()` after MPC outputs introduced grad-enabled tensors.

3. `RuntimeError: Trying to backward through the graph a second time` on second replan  
   Root cause: MPC viewer adaptation/handoff reused tensors that still carried autograd history from the previous plan.

## Changes

- [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - moved `from extension...` import below `sys.path` bootstrap block.
  - detached camera tensors before `.cpu().numpy()` conversion.
  - detached MPC viewer-result tensors in `_adapt_mpc_result_for_viewer`.
  - detached MPC state handoff tensors in `_mpc_state_from_reference_result`.

- [../../Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py](../../Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py)
  - new regression test to enforce import order: `sys.path` insertion must happen before `from extension...`.

- [../../Go2Pvcnn/tests/test_viewer_camera_detach.py](../../Go2Pvcnn/tests/test_viewer_camera_detach.py)
  - new regression test for `_update_camera` with `requires_grad=True` tensors.
  - new regression test ensuring `_adapt_mpc_result_for_viewer` returns detached tensors.

## Verification

- `pytest -q Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py Go2Pvcnn/tests/test_viewer_camera_detach.py` -> `3 passed`
- `python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py` -> pass
- exact user viewer command in `env_isaacsim` (`cuda:2`) -> process reached repeated MPC replan cycles (`cycle>=20`) without the three target crashes above; command kept running until external timeout/termination for log capture.

## Conclusion

The MPC viewer command now boots correctly, survives camera updates with grad-enabled tensors, and continues across repeated replans without autograd graph reuse failure.

## Follow-Up

- Remaining runtime output still includes Omniverse/Hydra warnings unrelated to this fix.
- If needed, add a lightweight runtime harness for fixed-duration viewer smoke to avoid manual timeout-based verification.

## Git Refs

- Baseline Ref: `130c635`
- Candidate Ref: working tree with viewer entrypoint/autograd replay fixes
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py](../../Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py)
  - [../../Go2Pvcnn/tests/test_viewer_camera_detach.py](../../Go2Pvcnn/tests/test_viewer_camera_detach.py)
