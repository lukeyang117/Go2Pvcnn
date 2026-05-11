# T116h Final Review And Authority

- timestamp: 2026-05-10 20:17 CST
- todo: [T100/T116h](../todo/T100-batched-together-planner-gpu-migration.md#t116h-final-integration-authoritative-rerun-review-and-noteslog-closure)
- result: pass

## Review Findings Handled

- Kept cross-window small obstacles in `CROSS_SMALL` even when a foot is already on `small`; the condition now remains a safety/diagnostic/cost issue instead of silently falling back to `CRUISE`.
- Hardened large runtime R6 beyond route selection: it now checks `status_sequence`, `safe_fallback_sequence`, touchdown semantics, foot collision, base/body/leg clearance, and non-center bypass route.
- Replaced hardcoded planner `direction_id=0` with GPU tensor command-axis IDs and surfaced them through runtime diagnostics; R7 checks lateral-left/right IDs.
- Renamed the too-high-small stop fallback test so it no longer implies a successful bypass route when all lateral candidates are hard-barriered.

## Verification

- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py -q -k "direction_id_reports_command_axis or too_high_small or cross_window_foot_on_small"` -> `5 passed, 32 deselected`.
- `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "together_viewer_adapter or runtime_plan_diagnostics_builds or grounded_crossing_runtime_sequence_report"` -> `3 passed, 27 deselected`.
- `timeout -s INT -k 20s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r6_large_runtime_bypass_direction_guard"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`.
- `timeout -s INT -k 20s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r7_lateral_runtime_no_opposite_direction_rejection_left_and_right"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`.
- Final deterministic/core/guardrail union: `PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batched_together_semantic_planner.py Go2Pvcnn/tests/test_batched_together_core.py Go2Pvcnn/tests/test_batched_together_guardrails.py -q` -> `54 passed`.
- Final runtime union: `timeout -s INT -k 20s 420s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k "test_r1_cruise_no_semantic_no_bypass or test_r2_small_cross_runtime_four_leg_success_all_command_directions or test_r3_small_cross_runtime_no_touchdown_on_small_all_directions or test_r4_small_cross_runtime_no_foot_path_collision_all_directions or test_r5_small_cross_runtime_no_base_body_leg_penetration_all_directions or test_r6_large_runtime_bypass_direction_guard or test_r7_lateral_runtime_no_opposite_direction_rejection_left_and_right"; code=$?; echo EXIT_CODE:$code; exit $code'` -> `EXIT_CODE:0`.
- `PYTHONPATH=Go2Pvcnn python -m py_compile Go2Pvcnn/extension/batched_together_planner/planner.py Go2Pvcnn/extension/batched_together_planner/parameterization.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py Go2Pvcnn/tests/test_batched_together_semantic_planner.py` -> pass.
- `git diff --check` -> pass.

## Residual Notes

- One old runtime command remains only inside `superseded_focused_commands`; it is historical/non-authoritative and no longer part of active T116 runtime acceptance.
- Runtime acceptance remains headless/output-based under `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim`.
