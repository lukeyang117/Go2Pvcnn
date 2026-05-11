# T115e `env_isaacsim` Headless Runtime Acceptance Cases

## Meta

- Time: `2026-05-08 22:29 +0800`
- Stage: `headless Isaac Lab runtime acceptance`
- Result: `pass`
- Todo: [T100/T115e](../todo/T100-batched-together-planner-gpu-migration.md#t115e-env_isaacsim-headless-runtime-acceptance-cases)

## Purpose

- Close `T115e` only.
- Run the approved `R1-R4` acceptance cases one by one under real `env_isaacsim` headless runtime.
- Require explicit terminal outcomes instead of the earlier sparse runtime-output observations.

## Scope

- Code changed only in:
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
- No together planner hot-path files changed in this leaf.
- No planner semantics were modified.

## Runtime Commands

All runtime commands used explicit timeout and exit-code reporting:

```bash
timeout -s INT -k 20s 240s bash -lc 'PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest <single-test> -q' >"$log" 2>&1
echo EXIT_CODE:$?
```

## Per-Case Results

### R1 `test_r1_small_cross_runtime_grounded`

- Result: `pass`
- Terminal evidence:
  - `EXIT_CODE:0`
- Log file:
  - `/tmp/t115e-r1-G58l.log`

### R2 `test_r2_small_bypass_runtime`

- Result: `pass`
- Terminal evidence:
  - `EXIT_CODE:0`
- Log file:
  - `/tmp/t115e-r2-Lh2Z.log`

### R3 `test_r3_rear_touchdown_airborne_regression`

- Result: `pass`
- Terminal evidence:
  - `EXIT_CODE:0`
- Log file:
  - `/tmp/t115e-r3-aeXS.log`

### R4 `test_r4_runtime_clear_requires_grounded_completion`

- Result: `pass`
- Terminal evidence:
  - `EXIT_CODE:0`
- Log file:
  - `/tmp/t115e-r4-qMcC.log`

## Minimal Runtime Union

- Command:
  - `PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn:/mnt/mydisk/lhy/testPvcnnWithIsaacsim /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest --noconftest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -k "test_r1_small_cross_runtime_grounded or test_r2_small_bypass_runtime or test_r3_rear_touchdown_airborne_regression or test_r4_runtime_clear_requires_grounded_completion or test_grounded_crossing_runtime_sequence_report_summarizes_acceptance_fields" -q`
- Timeout wrapper:
  - `timeout -s INT -k 20s 300s`
- Result:
  - `EXIT_CODE:0`
- Log file:
  - `/tmp/t115e-union-JbnK.log`

## Fixture / Helper Notes

- Added a targeted runtime helper around the existing `grounded_crossing` harness:
  - `plan_case_near_s4_anchor(...)`
  - `grounded_crossing_runtime_sequence(...)`
- Runtime helper defaults used for crossing-favorable sampling:
  - `x_offsets_m = (-0.18, 0.04, 0.28)`
  - `z_clearances = (0.65, 0.65, 0.65)`
- Runtime helper explicit parameters used for bypass / airborne-regression sampling:
  - `x_offsets_m = (-0.18,)`
  - `z_clearances = (0.85,)`
- No further parameter retuning was needed after explicit timeout-wrapped single-test runs started returning stable `EXIT_CODE:0`.

## Acceptance Coverage

- `R1` proves headless runtime sequence coverage for grounded `front_cross -> rear_follow -> clear`.
- `R2` proves runtime bypass selection path is executable under the targeted rear-not-groundable sample.
- `R3` proves the airborne rear-touchdown regression acceptance path executes cleanly and passes under the targeted runtime sample.
- `R4` proves `clear` acceptance remains covered under the runtime grounded-completion contract.
- Acceptance remains field-based and numeric; no viewer-image evidence is used.

## Caveats

- Full repository-wide pytest without `--noconftest` is still blocked by the historical `scripts.go2fp` dependency gap in `Go2Pvcnn/tests/conftest.py`.
- The timeout wrappers remain the recommended way to get explicit runtime end states on this machine, even though the four single tests and the minimal union all returned `EXIT_CODE:0`.

## Conclusion

- `T115e` is green on explicit real-runtime terminal outcomes.
- `R1-R4` each returned `EXIT_CODE:0` under `env_isaacsim` headless single-test execution with explicit timeout wrappers.
- Minimal runtime union also returned `EXIT_CODE:0`.

## Follow-up

- Carry this closure into `T115f` final authority / broader affected-union rerun if needed.
- Keep the same timeout-wrapped single-test pattern for future real-runtime acceptance collection on this machine.

## Git Refs

- Baseline Ref: `working tree after T115d runtime diagnostics surfacing`
- Candidate Ref: `working tree with T115e runtime acceptance helpers/tests and explicit timeout-wrapped env_isaacsim verification`
- Key Files:
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
