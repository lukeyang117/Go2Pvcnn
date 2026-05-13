# MPC Long Replan Foot Drift Reproduction

- Time: 2026-05-12 21:47 CST
- Purpose: reproduce the user-observed long-running MPC foot drift across command directions using IsaacLab headless runtime.
- Stage: `extension/batch_mpc_planner` viewer-style runtime diagnostics.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `e90e3a4`
- Candidate Ref: working tree with new reproduction test
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)

## Procedure

Added `test_mpc_runtime_long_replan_foot_drift_reproduction` under `Go2Pvcnn/tests/test_mpc_runtime_headless.py`.

The test is opt-in so normal focused selectors do not start IsaacLab unexpectedly:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 \
MPC_RUNTIME_LONG_DRIFT=1 \
MPC_LONG_DRIFT_CYCLES=120 \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_foot_drift_reproduction -s -q
```

Control checks:

```bash
python -m py_compile Go2Pvcnn/tests/test_mpc_runtime_headless.py
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_foot_drift_reproduction -q
```

The second control check passed as `1 skipped in 1.85s`, confirming the opt-in test does not launch IsaacLab unless enabled.

## Metrics

The IsaacLab headless test launched on `cuda:2`, attached the MPC trajectory manager, and completed six command directions over 120 replan/playback cycles.

| Command | rel_start | rel_end | drift | abs_drift | foot_err_mean | foot_err_last | foot_step_mean | dx_mean | dy_mean | dyaw_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 0.3978 | 0.4819 | +0.0841 | 0.0841 | 0.0683 | 0.0909 | 0.0072 | +0.2941 | +0.0000 | +0.0000 |
| backward | 0.4006 | 0.4728 | +0.0722 | 0.0722 | 0.0372 | 0.0405 | 0.0071 | -0.2103 | +0.0000 | +0.0000 |
| lateral_left | 0.4021 | 0.4073 | +0.0052 | 0.0052 | 0.0451 | 0.0465 | 0.0067 | +0.0000 | +0.2436 | +0.0000 |
| lateral_right | 0.3995 | 0.4318 | +0.0323 | 0.0323 | 0.0452 | 0.0465 | 0.0067 | +0.0000 | -0.2436 | +0.0000 |
| yaw_left | 0.4024 | 0.2852 | -0.1172 | 0.1172 | 0.0184 | 0.0164 | 0.0046 | +0.0004 | -0.0002 | -0.0204 |
| yaw_right | 0.3997 | 0.4300 | +0.0303 | 0.0303 | 0.0174 | 0.0196 | 0.0049 | +0.0004 | -0.0002 | +0.0206 |

Summary:

- `mean_abs_drift=0.0569`
- `max_name=yaw_left`
- `max_abs_drift=0.1172`
- reproduction threshold: `0.045 m`
- test result: pass as reproduction, because max drift exceeded threshold.

## Conclusion

The long-running foot drift is reproducible in the real IsaacLab headless path, not just in a local tensor/unit test.

The strongest current evidence remains structural: the MPC replan path reads current simulated foot positions at every replan and `build_nominal_trajectory()` rebuilds gait phase from fixed offsets each horizon. Forward/backward drift reproduces strongly, while lateral drift is smaller and yaw-left collapses foot-base radius most severely.

## Follow-Up

- Treat persistent gait/foothold memory as the next root-cause fix target:
  - manager-owned `stance_anchor_w`
  - `last_touchdown_w`
  - persistent gait phase
  - touchdown-only anchor update
- Keep this test as a reproduction oracle before changing planner behavior.
