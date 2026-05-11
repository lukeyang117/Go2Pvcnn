# T116i Nonzero Speed And Hard-Reason Implementation

## Purpose

Implement T116i: remove `beta=0` from nonzero-command K=5 candidate tables, add fixed-shape hard-reason tensors and all-hard ranking, and expose terminal/reporting reason summaries outside the planner hot path.

## Stage

`Go2Pvcnn/extension/batched_together_planner` core selection/cost/result path plus viewer/runtime diagnostics formatting.

## Related Todo

[T100/T116i](../todo/T100-batched-together-planner-gpu-migration.md#t116i-nonzero-speed-candidates-and-hard-reason-diagnostics)

## Command/Procedure

- RED: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "beta_tables"`
- RED: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "hard_reason or all_hard"`
- GREEN: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q`
- GREEN: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py -q`
- GREEN: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "format or runtime_summary or headless_output"`
- IsaacSim/headless-style: `timeout -s INT -k 20s 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py -q -k "headless or flat or small"`
- Compatibility subset: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "runtime_plan_diagnostics_builds_grounded_crossing_wrapper or grounded_crossing_runtime_sequence_report_summarizes_acceptance_fields"`
- Static/compile: `python3 -m py_compile ...`, `git diff --check -- ...`, and hot-path grep for `.cpu()`, `.tolist()`, `nonzero`, `argwhere`, `masked_select`, `numpy`, `np.`.

## Input Conditions

- Dirty worktree already contained prior T116/T117 edits and broad test cleanup deletions.
- Raw notes paths referenced by the repository rules were absent in this checkout, so available planner notes, todo branch, design, and plan were used.
- T116i worker write scope was limited to the approved planner, viewer diagnostics, and tests files.

## Key Metrics

- RED beta-table result: `1 failed, 3 deselected`; mismatch against new nonzero tables, including old `0.0` entries.
- RED hard-reason result: `3 failed, 1 passed`; missing `selected_hard_reason_mask` and `candidate_hard_rank_cost`.
- T116i focused GREEN: `6 passed`.
- Core/guardrail GREEN: `17 passed`.
- Formatting subset GREEN: `1 passed, 5 deselected`.
- env_isaacsim timeout-wrapped selected path GREEN after adding self-contained import path: `1 passed, 5 deselected`.
- Viewer diagnostics compatibility subset GREEN: `2 passed, 28 deselected`.
- Static guardrail grep: no matches in touched planner hot-path files.

## Result

Pass with scoped runtime coverage. The new deterministic tests and selected env_isaacsim headless-style path pass; broader real IsaacLab runtime cases in `test_viewer_runtime_diagnostics.py` still require the IsaacLab environment and were not rerun in full under the default Python.

## Conclusion

T116i behavior is implemented in the existing together planner path. Nonzero command beta tables no longer include `0.0`; zero-command hold remains; hard-reason masks/rank costs and selected candidate index are exposed; all-hard selection uses `candidate_hard_rank_cost` while keeping `ALL_INFEASIBLE`; viewer/test reporting can format selected and per-candidate hard reasons outside the hot path.

## Follow-Up

- A full real IsaacLab runtime union should be run by the coordinating agent if required for final authority beyond the selected env_isaacsim T116i command.
- Keep T117 test cleanup separate from T116i authority surfaces.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: uncommitted working tree
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner/types.py](../../Go2Pvcnn/extension/batched_together_planner/types.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/costs.py](../../Go2Pvcnn/extension/batched_together_planner/costs.py)
  - [../../Go2Pvcnn/extension/batched_together_planner/planner.py](../../Go2Pvcnn/extension/batched_together_planner/planner.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py](../../Go2Pvcnn/tests/test_t116i_nonzero_speed_hard_reasons.py)
