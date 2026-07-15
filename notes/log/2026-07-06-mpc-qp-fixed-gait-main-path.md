# 2026-07-06 MPC QP Fixed Gait Main Path

## Purpose

Connect the fixed alternating diagonal gait scaffold to the real `mpc_qp` continuous trajectory main path so the output contact schedule and stance anchors actually alternate.

## Stage

MPC-QP backend / T302v full SQP-QP trajectory backend.

## Related Todo

- [../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Commands:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_plan_segment_outputs_fixed_alternating_diagonal_contact_state -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_fixed_gait_keeps_stance_feet_anchored -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py
```

## Result

- RED confirmed `plan_segment_qp()` still emitted nominal contact state rather than fixed diagonal gait.
- RED confirmed stance feet were not anchored through the stance half of the horizon.
- GREEN:
  - `plan_segment_qp()` now builds `alternating_diagonal_gait_masks()` in continuous mode.
  - `decode_controls_to_result()` accepts an explicit `contact_state`.
  - Continuous `mpc_qp` output contact state is now fixed diagonal stance mask.
  - Stance foot samples are anchored for the stance half of each leg phase.
  - Added `qp_fixed_gait_active` diagnostic.
- Focused `mpc_qp`: `51 passed`.
- Current `mpc`/participation regression: `157 passed, 1 warning`.
- Pycompile: pass.

## Key Metrics

- `test_mpc_qp_backend.py`: `51 passed`.
- `test_mpc_rl_participation.py + test_batch_mpc_backend.py`: `157 passed, 1 warning`.

## Conclusion

The alternating gait issue is now addressed in code at the contact schedule and stance-anchor level for the opt-in continuous `mpc_qp` path. This is stronger than the previous scaffold-only state, but real viewer visual quality still depends on terrain and should be inspected on the hard tile.

## Follow-Up

- Re-run viewer on `terrain-row=8,col=12` to visually confirm foot timing and continuity after fixed gait anchoring.
- If stance anchoring causes hard-terrain foot discontinuities at phase switches, tune swing/stance transition smoothing through QP losses rather than adding candidate search or repair.

## Git Refs

- Baseline Ref: dirty workspace after fixed-shape SQP scaffold.
- Candidate Ref: dirty workspace after fixed gait main-path connection.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
