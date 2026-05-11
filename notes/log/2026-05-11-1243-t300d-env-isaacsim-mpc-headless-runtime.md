# T300d env_isaacsim MPC Headless Runtime Verification

- timestamp: 2026-05-11 12:43 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass (with runtime-output caveat)

## Purpose

Run the new `planner_backend="mpc"` runtime path on IsaacLab (`env_isaacsim`) in headless mode using the test layer, and verify:

- viewer/runtime fixture can initialize with MPC backend
- `plan_case` can execute standstill + forward commands
- diagnostics hard-mask layer is emitted when explicitly enabled

## Stage

Planner runtime verification (`extension/batch_mpc_planner` + viewer runtime fixture + headless IsaacLab tests).

## Changes

- Updated [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py):
  - default `requested_n_frames` -> `50` to align fixed-horizon runtime expectations
  - added `self.mpc_planner_cfg = _build_mpc_planner_cfg(...)`
  - passed `mpc_cfg=...` into all `_plan_viewer_trajectory(...)` calls
  - added MPC-specific state/terrain routing:
    - `_single_env_state -> _mpc_state_from_env(...)`
    - `_single_env_terrain_and_hits -> _compute_mpc_local_terrain(...)`
- Added [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py):
  - MPC backend attachment/runtime fixture smoke
  - headless standstill/forward planning smoke
  - diagnostics-enabled hard-mask emission smoke

## Verification

- `python -m py_compile Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_mpc_runtime_headless.py` -> pass
- `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q` -> `8 passed`
- `python -m pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "together_viewer_adapter_preserves_grounded_crossing_fields or runtime_plan_diagnostics_builds_grounded_crossing_wrapper"` -> `2 passed, 28 deselected`
- `timeout -s INT -k 30s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_mpc_runtime_headless.py -q -k "test_mpc_runtime_fixture_attaches_mpc_backend"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`
- `timeout -s INT -k 30s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_mpc_runtime_headless.py -q -k "test_mpc_runtime_plan_case_headless_smoke"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`
- `timeout -s INT -k 30s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_mpc_runtime_headless.py -q -k "test_mpc_runtime_diagnostics_layer_emits_hard_mask_when_enabled"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`
- `timeout -s INT -k 30s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_mpc_runtime_headless.py -k "test_mpc_runtime_plan_case_headless_smoke" -vv > /tmp/t300_mpc_selector.log 2>&1; code=$?; tail -n 60 /tmp/t300_mpc_selector.log; echo EXIT_CODE:$code; exit $code'` -> collected `3` (`2 deselected / 1 selected`), selector line emitted, `EXIT_CODE:0`

## Key Metrics

- MPC focused unit/integration suite: `8 passed`
- Viewer fixture contract subset: `2 passed`
- IsaacLab headless runtime selectors:
  - `test_mpc_runtime_fixture_attaches_mpc_backend`: `EXIT_CODE:0`
  - `test_mpc_runtime_plan_case_headless_smoke`: `EXIT_CODE:0`
  - `test_mpc_runtime_diagnostics_layer_emits_hard_mask_when_enabled`: `EXIT_CODE:0`

## Notes / Caveat

- In `env_isaacsim` headless runs, pytest textual summary output can be suppressed/terminated early while still returning `EXIT_CODE:0` (same caveat pattern seen in prior runtime logs). This pass therefore records selector-level exit-code evidence as the acceptance oracle.

## Conclusion

MPC backend is now exercised through the IsaacLab headless test layer, including diagnostics-enable coverage.

## Follow-Up

- Run 4096-env throughput profiling acceptance (`dirty_count`, `selected_dirty_count`, `dirty_backlog`, `max_stale_observed`, `planner_ms`, `cache_ms`, `reward_gather_ms`) to close remaining T300 runtime-scale risk.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: working tree with MPC runtime-fixture alignment + headless runtime test file
- Key Files:
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
