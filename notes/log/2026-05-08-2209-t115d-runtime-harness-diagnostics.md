# T115d `env_isaacsim` Runtime Harness And Diagnostics

## Meta

- Time: `2026-05-08 22:09 +0800`
- Stage: `headless Isaac Lab runtime harness diagnostics surfacing`
- Result: `pass with scoped runtime-output caveat`
- Todo: [T100/T115d](../todo/T100-batched-together-planner-gpu-migration.md#t115d-env_isaacsim-headless-isaac-lab-runtime-harness-and-diagnostics)

## Purpose

- Implement the `T115d` scope only.
- Extend the headless Isaac Lab runtime harness so the `together` runtime path exposes grounded rear-follow crossing diagnostics through a stable test-facing surface.
- Keep this leaf strictly at the harness/diagnostics layer rather than moving into `T115e` runtime acceptance outcomes.

## Scope

- Code changed only in:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
- No together planner semantics were changed in this leaf.
- No `R1-R4` runtime acceptance assertions were added in this leaf.

## Implementation Notes

- Extended `ViewerTrajectoryResult` so the `together` viewer/runtime adapter preserves planner-owned grounded-crossing diagnostics instead of dropping them at the viewer boundary.
- Added runtime harness surfacing in [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py):
  - `GroundedCrossingDiagnostics`
  - `RuntimePlanDiagnostics.grounded_crossing`
  - `RuntimePlanDiagnostics.grounded_crossing_summary`
- The new test-facing summary exposes these required metric names directly:
  - `state_mode`
  - `small_strategy_outcome`
  - `front_touchdown_ground_gap`
  - `rear_touchdown_ground_gap`
  - `touchdown_on_small_count`
  - `front_foot_small_collision_count`
  - `rear_foot_small_collision_count`
  - `base_small_penetration_count`
  - `base_path_crosses_small_flag`
- Fixed an existing runtime fixture mismatch unrelated to grounded-crossing semantics:
  - playback authoritative readback now converts robot joint order back to planner joint order before numeric comparison
  - this resolved the focused regression in `test_viewer_playback_matches_reference_frame_numeric`

## Verification

### TDD red

- Command:
  - `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -k "preserves_grounded_crossing_fields or builds_grounded_crossing_wrapper" -q`
- Initial result before implementation:
  - `2 failed`
- Initial failure surface:
  - `_adapt_together_result_for_viewer(...)` dropped grounded-crossing fields
  - `RuntimePlanDiagnostics` exposed no grounded-crossing wrapper

### Focused surfacing unit path

- Command:
  - `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -k "preserves_grounded_crossing_fields or builds_grounded_crossing_wrapper" -q`
- Result:
  - `2 passed`

### Focused real runtime grounded-crossing surfacing

- Command:
  - `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -k "grounded_crossing" -q`
- Result:
  - `..`
- Coverage from this focused path:
  - together headless runtime `plan_case("forward")` exposes `grounded_crossing`
  - required metric names are present on `grounded_crossing_summary`

### Affected playback regression

- Command:
  - `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py::test_viewer_playback_matches_reference_frame_numeric -q`
- Result:
  - `.`
- Diagnostic readback used during triage:
  - root readback delta max was about `3.3e-07`
  - joint readback delta max was about `3.09` before the joint-order normalization fix

### Broader affected runtime file

- Command:
  - `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q`
- Result:
  - focused reruns progressed through the previously failing playback point and no new regression was reproduced during the targeted reruns above
  - however the full file still exhibits the repository's known headless runtime-output/teardown caveat: output is sparse and final summary / clean exit reporting is not consistently surfaced in this environment

## Acceptance Coverage

- headless runtime harness now exposes the required grounded-crossing diagnostics at runtime plan boundaries:
  - covered by the new fixture wrapper and focused real-runtime tests
- metric source stays planner-owned and stable:
  - `TogetherPlannerResult -> ViewerTrajectoryResult -> GroundedCrossingDiagnostics`
- no viewer-image or manual visual assertion is required for this leaf:
  - all new acceptance evidence is field-based and numeric

## Caveats

- Repository-wide runtime pytest without `--noconftest` remains blocked by the existing `Go2Pvcnn/tests/conftest.py` import of missing `scripts.go2fp`.
- The full `test_viewer_runtime_diagnostics.py` file remains subject to the same headless Isaac runtime-output/teardown caveat already seen in prior logs:
  - focused surfacing and targeted regression paths are green
  - the broader file-level subset does not always return a clean final summary line promptly in this environment
- This log should be treated as `T115d` harness completion, not as `T115e` runtime acceptance for `R1-R4`.

## Conclusion

- `T115d` is implemented and verify-ready.
- The headless runtime harness now exposes the grounded rear-follow crossing diagnostics required by the approved design.
- Remaining `T115` work is runtime acceptance logic and final carry-forward authority, not additional harness surfacing.

## Follow-up

- Continue with `T115e` next using `grounded_crossing` / `grounded_crossing_summary` as the stable runtime-facing diagnostics surface.
- Keep the `--noconftest` focused runtime path as the standing verification route until the historical `scripts.go2fp` dependency gap is addressed separately.

## Git Refs

- Baseline Ref: `working tree after T115c deterministic fixture coverage with unrelated dirt preserved`
- Candidate Ref: `working tree with T115d runtime diagnostics surfacing and focused env_isaacsim verification`
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
