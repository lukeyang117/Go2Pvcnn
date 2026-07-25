# Joint MPC Cross Flat Metrics and Stance Follow-up

- Purpose: verify that small-obstacle crossing is gated by the complete flat/common metric set and investigate the remaining stance-grounding behavior.
- Stage: final perceptive-kinematic plan Task 17.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Changes

- Restored `build_nominal()` to one primary ranked selection, one preview selection, and one `_build_nominal_once()` call. The incomplete multi-attempt path had referenced undefined `preview_choices`.
- Kept the active crossing commitment selector override and exact nominal safety checks.
- Confirmed the small report uses `FINAL_METRIC_IDS`, which includes all `COMMON_METRICS`: stance ground gap/penetration, forbidden semantics, slip/anchor/stationary, swing surface, joint, root, lifecycle, map, KKT, publication, and validity metrics.
- Did not keep forced continuing-stance z reprojection. It reduced the short-trace gap to zero but reduced publication/validity to `2.7%`; the cached stance z and published IK ownership must be redesigned together.

## Verification

```text
3 passed in 6.63s
```

Focused checks covered the continuing-stance nominal contract, the small common-metric gate, and the flat-to-small metric registry relation. The nominal package had previously returned to `43 passed` after the single-nominal cleanup.

Short CUDA controlled trace before the rejected z reprojection showed:

- stance ground gap `0.00669m`, penetration `0.0m`;
- stance XY slip `4.85e-7m`, anchor residual `5.97e-7m`, stationary ratio `1.0`;
- publication and validity `1.0` for the short window.

The controlled 144-step crossing still fails strict crossing, so Task 17 is not complete. The failed behavior must be diagnosed at the published stance/IK ownership boundary without weakening common thresholds.

- Current Work Ref: `work/joint-mpc-kinematic`
- Candidate Ref: uncommitted working tree
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`, `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py`
