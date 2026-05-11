# T116i Main Review And Final Verification

## Purpose

Record the coordinating main-agent review after worker implementation and review-fix loops for T116i. This verifies that the final working tree covers nonzero-speed candidate tables, all-hard reason/rank diagnostics, production viewer output, and small-obstacle headless/runtime coverage.

## Stage

`Go2Pvcnn/extension/batched_together_planner` result/cost/selection diagnostics plus `Go2Pvcnn/extension/viz/go2_foostep_planner.py` production viewer reporting and `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py` acceptance coverage.

## Related Todo

[T100/T116i](../todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics)

## Command/Procedure

- Main review of worker output and diff for T116i scope.
- Read-only review subagent over T116i spec compliance.
- Fresh local deterministic/unit verification.
- Fresh timeout-wrapped `env_isaacsim` headless command.
- Residual process check for lingering Isaac/pytest processes.

Commands:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py -q
python3 -m py_compile Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py
git diff --check -- Go2Pvcnn/extension/batched_together_planner/types.py Go2Pvcnn/extension/batched_together_planner/costs.py Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "headless or flat or small" --collect-only
timeout -s INT -k 20s 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "headless or flat or small"
ps -eo pid,ppid,stat,etime,cmd | rg 'env_isaacsim|pytest|go2_foostep_planner|AppLauncher|isaac|omni|kit'
```

## Input Conditions

- Worker implementation had already added planner hard-reason tensors/rank costs and nonzero beta tables.
- Main review found two blockers: production viewer did not print hard-reason diagnostics, and the new T116i file did not include a small-obstacle headless/runtime crossing test.
- Worker review-fix added production `[Viewer][Plan]` hard-reason formatting and a small-obstacle all-direction runtime test.
- Main agent did not edit production code or test code; only notes/todo/log were updated by the controller.

## Key Metrics

- T116i file: `7 passed, 1 skipped`.
- T116i focused subset: `4 passed, 4 deselected`.
- Core/guardrail regression: `17 passed`.
- `py_compile`: exit `0`.
- `git diff --check`: exit `0`.
- Hot-path prohibited-pattern grep over `types.py`, `costs.py`, and `planner.py`: no matches.
- `env_isaacsim --collect-only -k "headless or flat or small"`: `2/8 tests collected (6 deselected)`, selecting flat nonzero and small-obstacle headless/runtime tests.
- Timeout-wrapped `env_isaacsim -k "headless or flat or small"`: exit `0`; quiet output emitted `.` only because Isaac app shutdown suppressed the full pytest summary.
- Process cleanup check: no lingering `env_isaacsim`, pytest, `go2_foostep_planner`, AppLauncher, Isaac, omni, or kit process matched after the timeout-wrapped run.
- Read-only review subagent result: `APPROVE`, no blocking T116i findings.

## Result

Pass with a runtime-output caveat. The final T116i working tree satisfies the approved T116i scope: nonzero commands cannot choose `beta=0` from the K=5 candidate tables, zero-command hold remains, all-hard selection exposes and uses fixed-shape hard-reason/rank tensors, production viewer plan lines print hard-reason details when needed, and the new T116i test surface includes small-obstacle crossing for forward, backward, lateral-left, and lateral-right commands under `env_isaacsim`.

## Conclusion

T116i is closed from the controller perspective. The only caveat is output formatting from the IsaacLab runtime command: pytest's final summary is suppressed after app shutdown, so authority is based on selected-test collection, exit code `0`, and no lingering runtime processes.

## Follow-Up

- Keep T117 cleanup separate from T116i authority surfaces.
- If future runtime debugging needs exact per-case text output, prefer `-vv -k "headless_small"` or a fixture-level report printer because quiet pytest output may be swallowed during Isaac app shutdown.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py](../../Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py)
