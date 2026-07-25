# Joint MPC Stance Grounding Metric Follow-up

- Purpose: verify that small/cross inherits flat stance metrics and investigate the remaining stance center-height gap.
- Stage: final perceptive-kinematic plan Task 17.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Result

- `COMMON_METRICS` is included by `FINAL_METRIC_IDS`, so small/cross evaluates stance ground gap, penetration, forbidden semantics, anchor/slip, continuity, joint, root, lifecycle, map, KKT, publication, and validity metrics.
- A controlled short trace reports approximately `stance_ground_gap=0.00669m`, `stance_ground_penetration=0`, `stance_xy_slip_max=1.39e-6m`, and `stance_anchor_residual=1.10e-6m` before the unrelated long crossing failure.
- Directly projecting the foot center to `height + foot_contact_offset (0.0221m)` is not a valid fix: the current swept safety model treats the foot center with its collision radius and rejects the nominal, causing publication to stop.
- The attempted current-stance ownership rewrite also failed all 24 flat nominal phases and was reverted. Existing anchor ownership remains unchanged.

## Verification

```text
test_nominal.py::test_only_first_optimize_after_reset_is_cold: passed
test_nominal.py::test_continuing_stance_uses_full_xyz_persistent_anchor_over_horizon_segment: passed
test_nominal.py::test_flat_nominal_is_hard_safe_for_all_24_start_phases: passed
3 passed in 19.56s
```

## Conclusion

The common flat metric gate is wired and tested. Stance center-height alignment remains an open geometry-contract issue; do not solve it by weakening collision filters or forcing z projection. The long controlled crossing remains red for its independent swing-foot crossing event.

- Current Work Ref: `work/joint-mpc-kinematic`
- Candidate Ref: uncommitted working tree
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`, `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`, `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`, `Go2Pvcnn/extension/joint_mpc_rti/terrain/swept_safety.py`
