# MPC Long Replan Variant Sweep

- Time: 2026-05-12 22:43 CST
- Purpose: compare five proposed MPC drift-improvement directions against the same long IsaacLab headless replan/playback metric used for the prior drift reproduction.
- Stage: `extension/batch_mpc_planner` viewer-style IsaacLab runtime diagnostics.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `e90e3a4`
- Candidate Ref: working tree with opt-in long drift sweep test
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)

## Procedure

Added `test_mpc_runtime_long_replan_variant_sweep` under `Go2Pvcnn/tests/test_mpc_runtime_headless.py`.

The test is opt-in:

```bash
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -q
# 1 skipped in 1.86s
```

Control checks:

```bash
python -m py_compile Go2Pvcnn/tests/test_mpc_runtime_headless.py
```

IsaacLab smoke and full sweep:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=20 \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q

MPC_TEST_DEVICE=cuda:2 MPC_RUNTIME_LONG_DRIFT_SWEEP=1 MPC_LONG_DRIFT_CYCLES=120 \
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_long_replan_variant_sweep -s -q
```

Both IsaacLab runs completed with pytest `.` / exit code `0`.

## Variants

| Variant | Direction Tested | Implementation Scope |
| --- | --- | --- |
| `baseline` | current planner behavior | no behavior change |
| `dir1_stance_anchor_proxy` | manager maintains stance anchors | test-layer proxy clamps prior-contact input feet to persistent anchors |
| `dir2_phase_continuity` | manager maintains continuous gait phase | monkeypatch shifts nominal leg phase offsets across replans |
| `dir3_anchor_nominal_proxy` | nominal generation uses anchor + swing target | monkeypatch replaces nominal contact frames with persistent anchors |
| `dir4_stronger_stance_loss` | stronger stance/root-frame losses | test-layer config weights for stance/root-frame/touchdown terms |
| `dir5_diagnostics_only` | diagnostics/fail-fast metrics | enables diagnostics and emits extra drift diagnostics without intended behavior change |

## Full 120-Cycle Summary

| Variant | mean_abs_drift | delta vs baseline | max_name | max_abs_drift |
| --- | ---: | ---: | --- | ---: |
| baseline | 0.0634 | +0.0000 | yaw_left | 0.1474 |
| dir1_stance_anchor_proxy | 0.0627 | -0.0007 | yaw_left | 0.1474 |
| dir2_phase_continuity | 0.0666 | +0.0033 | yaw_left | 0.1709 |
| dir3_anchor_nominal_proxy | 0.0793 | +0.0159 | lateral_left | 0.1247 |
| dir4_stronger_stance_loss | 0.0741 | +0.0107 | yaw_right | 0.1090 |
| dir5_diagnostics_only | 0.0598 | -0.0035 | yaw_left | 0.1424 |

## Direction-Level Drift

| Command | baseline | dir1 | dir2 | dir3 | dir4 | dir5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | +0.0842 | +0.0841 | +0.0841 | +0.1108 | +0.0670 | +0.0841 |
| backward | +0.0758 | +0.0720 | +0.0720 | +0.1059 | +0.0661 | +0.0720 |
| lateral_left | +0.0052 | +0.0052 | +0.0052 | +0.1247 | +0.0372 | +0.0052 |
| lateral_right | +0.0323 | +0.0323 | +0.0323 | +0.0684 | +0.0660 | +0.0323 |
| yaw_left | -0.1474 | -0.1474 | -0.1709 | -0.0645 | -0.0991 | -0.1424 |
| yaw_right | +0.0353 | +0.0353 | +0.0353 | -0.0014 | +0.1090 | +0.0230 |

## Conclusion

- `dir1_stance_anchor_proxy` is effectively neutral. Input-level clamping does not fix yaw-left collapse or forward/backward drift, so the anchor state has to enter the planner's contact/touchdown semantics rather than only the next replan input state.
- `dir2_phase_continuity` alone does not help. It worsened `yaw_left`, which suggests phase continuity must be coupled with touchdown/anchor ownership, not applied as an isolated nominal phase offset.
- `dir3_anchor_nominal_proxy` contains useful signal but is not directly usable. It improves yaw drift strongly, but makes linear/lateral motion and playback error worse because the stance nominal is hard-replaced over whole contact frames.
- `dir4_stronger_stance_loss` is direction-dependent and not safe as a global weight increase. It improves `forward`, `backward`, and `yaw_left`, but worsens lateral and `yaw_right`.
- `dir5_diagnostics_only` behaves close to baseline and is useful as instrumentation, not as a fix.

## Follow-Up

The next implementation direction should be a real manager-owned gait/foothold memory design:

- update `stance_anchor_w` only on touchdown / stable contact
- feed anchors into nominal/loss through contact-aware masks, not full stance-frame replacement
- keep gait phase continuous but reset or realign it on contact events
- make stance-anchor and root-frame losses direction-adaptive, especially for yaw vs lateral commands
