# T116g env_isaacsim Headless Runtime Diagnostics

- timestamp: 2026-05-10 19:53 CST
- todo: [T100/T116g](../todo/T100-batched-together-planner-gpu-migration.md#t116g-env_isaacsim-headless-runtime-diagnostics-and-acceptance-tests)
- result: pass

## Changes

- Migrated viewer runtime diagnostics from stale `front_cross/rear_follow/clear` state assumptions to T116 mode-first output metrics.
- Surfaced `status`, `feasible`, `safe_fallback`, selected beta/route, command-direction violation, cross-small success, per-leg touchdown/semantic/gap/clearance, and base/body/leg clearance through the viewer adapter and runtime fixture.
- Added command-relative runtime placement so small/large obstacles are tested in the command corridor for forward, backward, lateral-left, and lateral-right.
- Added R1-R7 headless acceptance tests for cruise/no-semantic, small crossing, touchdown-on-small, foot path collision, base/body/leg penetration, large bypass, and lateral direction rejection.

## Verification

- TDD red: local and `env_isaacsim` collection initially failed on stale `STATE_BYPASS` imports from the old runtime schema.
- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "together_viewer_adapter or runtime_plan_diagnostics_builds or grounded_crossing_runtime_sequence_report"` -> `3 passed, 27 deselected`.
- `timeout -s INT -k 20s 120s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_together_viewer_adapter_preserves_grounded_crossing_fields or test_runtime_plan_diagnostics_builds_grounded_crossing_wrapper or test_grounded_crossing_runtime_sequence_report_summarizes_acceptance_fields"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `3 passed, 26 deselected`, `EXIT_CODE:0`.
- `timeout -s INT -k 20s 240s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r1_cruise_no_semantic_no_bypass"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`.
- `timeout -s INT -k 20s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r2_small_cross_runtime_four_leg_success_all_command_directions or test_r3_small_cross_runtime_no_touchdown_on_small_all_directions or test_r4_small_cross_runtime_no_foot_path_collision_all_directions or test_r5_small_cross_runtime_no_base_body_leg_penetration_all_directions"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`.
- `timeout -s INT -k 20s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r6_large_runtime_bypass_direction_guard or test_r7_lateral_runtime_no_opposite_direction_rejection_left_and_right"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`.
- Final same-state runtime union: `timeout -s INT -k 20s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r1_cruise_no_semantic_no_bypass or test_r2_small_cross_runtime_four_leg_success_all_command_directions or test_r3_small_cross_runtime_no_touchdown_on_small_all_directions or test_r4_small_cross_runtime_no_foot_path_collision_all_directions or test_r5_small_cross_runtime_no_base_body_leg_penetration_all_directions or test_r6_large_runtime_bypass_direction_guard or test_r7_lateral_runtime_no_opposite_direction_rejection_left_and_right"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`.
- `PYTHONPATH=Go2Pvcnn python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py` -> pass.
- `rg -n "STATE_BYPASS|STATE_CLEAR|STATE_FRONT_CROSS|STATE_REAR_FOLLOW|front_cross|rear_follow|clear_requires_grounded_completion|runtime_bypass_selected_when_rear_not_groundable" Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py` -> no matches.

## Notes

- Acceptance is headless output-based under `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`; no screenshot or visual-only evidence was used.
- Runtime fixture/test CPU summaries remain outside the planner training hot path.
- T116h remains for final review and final-code-state deterministic/runtime authority.
