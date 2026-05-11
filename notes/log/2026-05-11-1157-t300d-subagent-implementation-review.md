# T300d Subagent Implementation Review And Integration

## Purpose

Integrate and main-agent-review the subagent-driven `batch_mpc_planner` implementation, then verify compile/test evidence for the new `planner_backend="mpc"` path.

## Stage

Planner implementation + viewer/factory wiring + focused contract tests for:

- `Go2Pvcnn/extension/batch_mpc_planner/*`
- `Go2Pvcnn/extension/trajectory_manager_factory.py`
- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`
- `Go2Pvcnn/tests/test_batch_mpc_backend.py`
- `docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md`

## Related Todo

[T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)

## Command/Procedure

- Pulled subagent outputs and reviewed worker claims against local diffs.
- Fixed manager edge cases:
  - no-empty-subset replan path
  - step-token `None` refresh behavior
  - fixed-budget dirty scheduling keeps full-shape cache semantics.
- Extended config plumbing:
  - `mpc_planner_cfg` object override support
  - per-loss/per-diagnostics/per-runtime tunables via task cfg bridge.
- Enabled configurable touchdown event cap contract and relaxed result shape check to `[B,4,E,3]`.
- Added/extended tests for:
  - factory/backend selection
  - manager cache refresh and no-dirty step advance
  - configurable touchdown event cap
  - loss/diagnostics override mapping
  - CUDA path plan execution.
- Updated viewer backend routing to include `mpc` and adapted MPC result diagnostics/status rendering.
- Synced design doc naming (`MpcTrajectoryManager`, `batch_mpc_planner` path).

## Input Conditions

- Subagent execution was requested explicitly by user.
- Workspace is dirty with many unrelated historical changes; this pass touched only T300d-relevant files.

## Verification Commands

- `python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py`
- `python -m py_compile Go2Pvcnn/extension/trajectory_manager_factory.py Go2Pvcnn/extension/trajectory_contracts.py $(find Go2Pvcnn/extension/batch_mpc_planner -name '*.py' | sort)`
- `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q`
- `git diff --check -- Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/tests/test_batch_mpc_backend.py docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md`

## Key Metrics

- Focused backend tests: `8 passed` (`test_batch_mpc_backend.py`)
- Compile checks: target files passed (`py_compile` exit `0`)
- CUDA test path: included in focused suite (passed in current environment)
- Hard diagnostics path: enabled and asserted in focused tests

## Result

Pass (implementation + focused verification).  
The new MPC backend path is wired into factory and viewer selection, with config-first tunables and subagent-reviewed integration updates.

## Conclusion

T300d implementation is functionally integrated and locally verified at unit/focused-integration level.  
Remaining acceptance risk is full `env_isaaclab` runtime throughput/quality validation at 4096 environments.

## Follow-Up

- Run GPU runtime acceptance on `env_isaaclab` with real async command/reset patterns.
- Capture planner dirty-budget counters and per-step timing metrics under 4096 env workload.
- Confirm viewer-side MPC diagnostics readability during obstacle-crossing scenarios.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: working tree with T300d integration edits
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md](../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md)
