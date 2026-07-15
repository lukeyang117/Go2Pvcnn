# 2026-07-06 MPC QP Fixed-Shape SQP Scaffold

## Purpose

Implement the first code pass for the full SQP/QP `mpc_qp` direction: remove candidate/search behavior from the continuous solver main path, add fixed-shape differentiable field/gait/variable/QP assembly scaffolding, and keep the default `mpc` backend isolated.

## Stage

MPC-QP backend / T302v full SQP-QP trajectory backend.

## Related Todo

- [../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Commands:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q -k 'differentiable_fields or gait_masks or variable_layout or assembly_returns or no_candidate'
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py
git diff --check
rg -n "_repeat_terrain_for_candidates|candidate_xy|best_idx|best_score|scales = torch.tensor|best_scale_idx|semantic_repair|fixed_repair_offsets" Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py Go2Pvcnn/extension/batch_mpc_qp_planner/fields.py Go2Pvcnn/extension/batch_mpc_qp_planner/gait.py Go2Pvcnn/extension/batch_mpc_qp_planner/variables.py Go2Pvcnn/extension/batch_mpc_qp_planner/qp_assembly.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:0 --num_envs 1 --terrain task --planner-backend mpc_qp --n-frames 25 --plan-dt 0.02 --terrain-row 8 --terrain-col 12
```

## Result

- RED confirmed missing `fields.py`, `gait.py`, `variables.py`, `qp_assembly.py`, and candidate/search tokens in `solver.py`.
- GREEN:
  - Added differentiable field scaffold with `height/grad_h`, smoothed semantic risk/gradient, and roughness/gradient queries.
  - Added fixed alternating diagonal gait masks.
  - Added fixed QP variable layout with `touchdown_xy` indices and slack variables.
  - Added fixed-shape QP matrix assembly scaffold.
  - Removed endpoint scale candidate selection and fixed-offset touchdown candidate selection from `solver.py`.
  - Replaced touchdown update with a bounded field-gradient QP-style step.
  - Changed FK/readback behavior to fixed residual refinement; quality failures now improve by increasing `qp_iterations`, not by endpoint candidates.
- Focused `mpc_qp`: `49 passed`.
- Current `mpc`/participation regression: `157 passed, 1 warning`.
- Pycompile: pass.
- `git diff --check`: pass.
- Candidate/search token `rg`: no matches in the new continuous solver/scaffold files.
- Real IsaacLab viewer command reached environment setup, attached `[Viewer] Attached mpc_qp trajectory manager`, applied `[Viewer] Terrain tile override: row=8 col=12 origin=(+28.000, +20.000, +2.054)`, entered playback loop, ran without planner exception for about 30 seconds, then exited cleanly on Ctrl-C.

## Key Metrics

- `test_mpc_qp_backend.py`: `49 passed`.
- `test_mpc_rl_participation.py + test_batch_mpc_backend.py`: `157 passed, 1 warning`.
- No hard repair/candidate tokens in the checked continuous main-path files.
- Required viewer terrain `row=8,col=12`: startup/attach/tile override/passive playback-loop smoke passed; detailed visual trajectory quality was not scored automatically in this command.

## Conclusion

The code now has the first fixed-shape full-SQP/QP scaffold for `mpc_qp`, and the continuous solver no longer uses endpoint candidates or fixed-offset touchdown selection as its main path. The default `mpc` regression remains green. The required row/col viewer command starts and runs with `mpc_qp`; automated visual-quality metrics still need a dedicated probe if we want pass/fail numbers beyond startup and no-exception evidence.

## Follow-Up

- Add an automated row/col hard-terrain probe for `terrain-row=8,col=12` if visual quality needs numeric gating rather than manual viewer inspection.
- Continue replacing legacy repair diagnostics and old repair-named config fields with field/loss/QP naming where they are still only historical compatibility counters.

## Git Refs

- Baseline Ref: dirty workspace before this pass.
- Candidate Ref: dirty workspace after local edits; many `mpc_qp` files remain untracked in this workspace.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/fields.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/gait.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/variables.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/qp_assembly.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
