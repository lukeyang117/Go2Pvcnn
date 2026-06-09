# T302p Low-Small FK Loss Wiring And Regression

## Purpose

Record the T302p continuation that fixed the low-small FK semantic regression without adding a new loss. The pass also records the remaining flat-direction failure so T302p is not closed prematurely.

## Stage

MPC semantic policy evaluation / parametric batch MPC sampled loss path.

## Related Todo

- [../todo/T302p-mpc-command-frame-alignment-plan.md](../todo/T302p-mpc-command-frame-alignment-plan.md)
- Previous failure log: [2026-06-06-2317-t302p-real-acceptance-failures.md](2026-06-06-2317-t302p-real-acceptance-failures.md)

## Changes

- Wired existing `cfg.losses.ik_fk_residual.weight` into existing `parametric_trajectory_fk_consistency_loss()`.
- Wired existing `cfg.losses.kinematics.weight` and `joint_limit_margin_rad` into the parametric sampled loss path as `parametric_joint_limit`.
- Changed existing `parametric_fk_body_leg_collision_loss()` aggregation from mean-only to mean + worst for foot/knee/shank/root/underbody collision costs, so sparse FK contact is not diluted across long horizons.
- No new loss term was added to optimize a new metric; the changes make already configured loss families effective.

## Commands

Focused RED/GREEN and regression:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_parametric_joint_limit_uses_existing_kinematics_weight_and_margin -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_fk_body_leg_collision_keeps_sparse_foot_collision_salient_across_horizon -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git diff --check
```

Real low-small regression:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 420s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py \
  --device cuda:0 \
  --variants parametric_v1 \
  --requested-n-frames 300 \
  --warmup-steps 6 \
  > tmp/t302p-command-frame/low_small_after_joint_limit_fk_worst_gpu0.jsonl 2>&1
```

Flat-left tracking smoke:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 240s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking --headless --device cuda:0 --num-envs 1 \
  --num-rounds 1 --max-steps 120 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "0.0 1.0 0.0" \
  --terrain-rows 0 --terrain-cols 0 \
  --output-dir tmp/t302p-command-frame/left_120_after_low_small_fix
```

## Key Metrics

Local:

- Focused RED for missing `parametric_joint_limit`: failed with `KeyError: 'parametric_joint_limit'`.
- Focused RED for sparse FK collision dilution: long-horizon loss was `1/10` of short-horizon loss.
- Focused GREEN: `3 passed`.
- Backend + parametric regression: `150 passed, 1 warning`.
- Pycompile: exit `0`.
- `git diff --check`: exit `0`.

Low-small real regression after fix:

- Output: `tmp/t302p-command-frame/low_small_after_joint_limit_fk_worst_gpu0.jsonl`
- Rows: `5`
- `max_fk_semantic_collision_count = 0`
- `max_fk_semantic_collision_rate = 0.0`
- `max planned_vs_fk_foot_error_crossing_leg_max_m = 0.04200904071331024`
- `max planned_vs_fk_foot_error_all_max_m = 0.131496399641037`
- Per command collision counts:
  - `forward_v050 = 0`
  - `lateral_v050 = 0`
  - `diagonal_v050 = 0`
  - `mixed_yaw_v050 = 0`
  - `yaw100 = 0`

Flat-left tracking smoke after fix:

- Output: `tmp/t302p-command-frame/left_120_after_low_small_fix/2026-06-07_11-02-14-594186`
- Exit: `0`
- `command_body_match_max_abs_error = 0.0`
- `planned_root_direction_cosine = 0.9761884212493896`
- `planned_root_lateral_ratio = 0.21692466735839844`
- `reference_valid_ratio = 1.0`

## Result

Low-small semantic compatibility hard gates pass on GPU0/env_isaacsim with default `parametric_v1` after the loss wiring and existing loss aggregation fix.

T302p is still not complete because the flat-left tracking smoke still shows high root lateral ratio (`0.2169`) even with command-source equality. Flat all-direction root and per-leg direction acceptance remains open.

## Follow-Up

- Keep T302p.2 active for flat direction debugging.
- Treat low-small FK semantic collision as fixed/regression-guarded for this pass.
- Do not claim full T302p acceptance until flat no-obstacle all-direction root XY and moving-leg XY direction metrics pass.

## Git Refs

- Baseline Ref: local dirty T302p command-frame implementation after 2026-06-06 failure run.
- Candidate Ref: local dirty worktree with existing FK/kinematics loss wiring and FK collision aggregation fix.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/parametric_losses.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`
