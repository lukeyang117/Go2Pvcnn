# T300d 4096 Runtime Counters Attempt (env_isaacsim Headless)

- timestamp: 2026-05-11 13:18 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: partial pass with 4096 runtime blocker

## Purpose

Continue T300d runtime acceptance by adding planner runtime counters and attempting true headless IsaacLab verification for large parallel scale (target `num_envs=4096`).

## Stage

MPC runtime instrumentation + test-layer expansion + IsaacLab headless verification attempts.

## Changes

- Added runtime counter config toggles in [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py):
  - `mpc_diagnostics_emit_runtime_counters`
  - `mpc_diagnostics_profile_cuda_sync`
- Added runtime counter collection in [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py):
  - exposed `runtime_counters()`
  - emits `dirty_count`, `selected_dirty_count`, `dirty_backlog`, `max_stale_observed`, `planner_ms`, `cache_ms`
- Added unit coverage in [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py):
  - verifies runtime counter emission contract
- Added headless runtime tests in [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py):
  - existing MPC runtime smoke retained
  - new 4096 gate test `test_mpc_runtime_4096_headless_dirty_budget_counters`
  - test is opt-in via `MPC_RUNTIME_4096=1`
- Added large-env PhysX capacity tuning hook in [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py):
  - `_configure_large_runtime_physx_buffers()` for `num_envs>=2048`

## Verification

- `python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_runtime_headless.py` -> pass
- `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q` -> `9 passed`
- baseline headless selector checks (from previous node) remain `EXIT_CODE:0` for 3 MPC runtime selectors.
- 4096 selector attempt:
  - `MPC_RUNTIME_4096=1 ... pytest --noconftest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k "test_mpc_runtime_4096_headless_dirty_budget_counters" -vv -s`
  - collected `4` (`3 deselected / 1 selected`)
  - runtime never reached test-body counter prints; Isaac process stayed in long runtime loop and finally shut down without pytest summary line.
- direct 512/4096 scripted counter sampling attempts in `env_isaacsim`:
  - no `MPC_*_COUNTERS_*` print observed
  - logs show high-scale PhysX/GPU instability (`foundLost* capacity errors`, `omni.physx.tensors` CUDA device-side assert).

## Key Metrics

- local focused backend suite: `9 passed`
- `env_isaacsim` 4096 selector: selected path reached, but runtime counters unavailable due startup/runtime instability.
- observed blocker signatures:
  - `PhysX ... increase PxGpuDynamicsMemoryConfig::foundLostPairsCapacity`
  - `omni.physx.tensors.plugin CUDA error: device-side assert triggered`
  - long datastore GC loop without test-body progress.

## Conclusion

Runtime counter instrumentation is implemented and unit-verified.  
True IsaacLab headless 4096 acceptance remains blocked by high-scale runtime instability in current environment/config path.

## Follow-Up

- keep 4096 test opt-in (`MPC_RUNTIME_4096=1`) and non-blocking for default runs.
- add a dedicated large-scale runtime profile (non-semantic/lightweight scene) for 4096 acceptance, separate from current semantic viewer fixture.
- re-run 4096 acceptance after scene/profile split and PhysX budget tuning to capture authoritative counter values.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: working tree with runtime counter instrumentation + 4096 gated runtime attempts
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
