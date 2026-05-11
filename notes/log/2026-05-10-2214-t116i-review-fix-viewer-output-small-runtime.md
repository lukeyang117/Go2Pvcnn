# T116i Review Fix: Viewer Output And Small Runtime Coverage

## Purpose

Address T116i review blockers: production `[Viewer][Plan]` output must print hard-reason diagnostics, and the new T116i test file must include a selected small-obstacle headless/runtime test.

## Stage

`Go2Pvcnn/extension/viz/go2_foostep_planner.py` production viewer reporting plus `Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py` runtime coverage.

## Related Todo

[T100/T116i](../todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics)

## Command/Procedure

- RED viewer output: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "production_viewer_plan_line"`
- RED/default small selector: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "headless_small"`
- GREEN T116i: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q`
- GREEN core/guardrail: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
- Required env_isaacsim: `timeout -s INT -k 20s 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "headless or flat or small"`
- Selection proof: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "headless or flat or small" --collect-only`
- Small runtime proof: `timeout -s INT -k 20s 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -vv -k "headless_small"`
- Static: `py_compile`, `git diff --check`, and hot-path grep over touched planner files.

## Input Conditions

- T116i implementation already had planner tensor diagnostics and test-fixture formatting.
- Production viewer still printed only backend/cycle/cmd/delta/dyaw/standstill/semantic.
- The required `-k "headless or flat or small"` command previously selected only one flat test.

## Key Metrics

- RED viewer output: failed with missing `_format_viewer_plan_line`.
- Default small selector: `1 skipped, 7 deselected` under default Python because `isaaclab.app` is unavailable.
- GREEN T116i default: `7 passed, 1 skipped`.
- GREEN core/guardrail: `17 passed`.
- env_isaacsim required selection: collection shows `2/8 tests collected (6 deselected)` for `headless or flat or small`.
- env_isaacsim required run: exit code `0`; quiet output emitted one dot because the Isaac app shutdown suppresses final pytest summary.
- env_isaacsim small-only proof: `collected 8 items / 7 deselected / 1 selected`, selected `test_t116i_headless_small_obstacle_crossing_runtime_all_directions`, exit code `0`.
- Hot-path grep across touched planner files returned no matches.

## Result

Pass with output caveat. Production viewer reporting now prints `status=...` and, when all-infeasible or selected hard reason is non-empty, appends `selected_hard_reasons=...`, `selected_hard_rank_cost=...`, `candidate_hard_rank=[...]`, and `candidate_hard_reasons=[...]` to the live `[Viewer][Plan]` line. The T116i runtime/headless selector now includes the flat test and the small-obstacle all-direction runtime test.

## Conclusion

The two review blockers are addressed without adding planner source files or changing planner hot-path CPU behavior.

## Follow-Up

- The env_isaacsim small runtime command exits `0`, but final pytest summary text is truncated/suppressed after Isaac app shutdown; selection and selected-test name were captured separately.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py](../../Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
