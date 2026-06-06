# T302p Command Frame Implementation

## Purpose

Record the T302p implementation and verification pass for aligning MPC command-frame interpretation with the public body/root-yaw command contract.

## Stage

MPC semantic policy evaluation / batch MPC command-frame runtime contract.

## Related Todo

- [../todo/T302p-mpc-command-frame-alignment-plan.md](../todo/T302p-mpc-command-frame-alignment-plan.md)
- Parent reproduction: [2026-06-06-1633-t302o-flat-forward-mpc-left-bias-reproduction.md](2026-06-06-1633-t302o-flat-forward-mpc-left-bias-reproduction.md)

## Procedure

- Added RED/static tests for:
  - root-yaw body command axes in `command_frame_axes()`
  - viewer MPC planning boundary keeping body-frame command
  - eval command-source diagnostics and planned direction metrics
- Implemented the coordinate-frame change:
  - `command_frame_axes(command_body, root_yaw)` rotates nonzero body command XY by root yaw.
  - MPC world-geometry heading uses now call `command_frame_axes()` with current root yaw.
  - Body-frame intent checks remain based on raw command XY speed, `vy_body`, and yaw-rate.
  - Viewer no longer pre-rotates the command before `plan_segment()`.
  - Eval records requested/policy/MPC command equality and planned root/leg direction metrics.
- Ran focused local tests, pycompile, diff check, and one short real IsaacLab smoke.

## Commands

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
pytest Go2Pvcnn/tests/test_viewer_reset.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_viewer_reset.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/scripts/mpc_policy_eval.py
git diff --check
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py --mode tracking --headless --device cuda:0 --num-envs 1 --num-rounds 1 --max-steps 5 --run-dir 2026-05-31_20-03-27 --checkpoint model_14000.pt --command-mode fixed --command "1.0 0.0 0.0" --output-dir logs/mpc_policy_eval/t302p_flat_forward_smoke
```

## Input Conditions

- Fixed forward command: `[1.0, 0.0, 0.0]`
- Public command contract: body/root-yaw frame `[vx_body, vy_body, yaw_rate]`
- Real smoke environment: GPU0, `env_isaacsim`, `num_envs=1`, `max_steps=5`
- Checkpoint: `run_dir=2026-05-31_20-03-27`, `checkpoint=model_14000.pt`

## Key Metrics

- Focused local suite: `184 passed, 1 warning`
- Pycompile: exit `0`
- `git diff --check`: exit `0`
- Real smoke output: `logs/mpc_policy_eval/t302p_flat_forward_smoke/2026-06-06_18-57-34-897751`
- Real smoke summary:
  - `command_body_match_max_abs_error = 0.0`
  - `requested_command_body = [[1.0, 0.0, 0.0]]`
  - `policy_command_body = [[1.0, 0.0, 0.0]]`
  - `mpc_input_command_body = [[1.0, 0.0, 0.0]]`
  - `planned_root_direction_cosine = 0.9988572597503662`
  - `planned_root_lateral_ratio = 0.04779195412993431`
  - `planned_per_leg_direction_cosine_xy = [0.9994088411331177, 0.9887953400611877, 0.9957641959190369, 0.9997162818908691]`
  - `planned_per_leg_lateral_ratio_xy = [0.03437824174761772, 0.14927777647972107, 0.0919436365365982, 0.023816389963030815]`
  - `reference_valid_ratio = 1.0`

## Result

Implementation and focused local verification passed. The short real IsaacLab smoke confirms command-source equality and root direction alignment for fixed forward command.

The full T302p acceptance is not closed because:

- only one command direction was smoke-tested in IsaacLab
- the run was only `5` steps
- one leg lateral ratio was `0.14927777647972107`, above the preferred `0.10` moving-leg threshold
- low-small semantic compatibility regression was not rerun

## Conclusion

The coordinate-frame code path is implemented and locally guarded. T302p still needs a longer eight-command flat direction acceptance run and low-small semantic compatibility regression before final closure.

## Follow-Up

- Run the eight-command flat no-obstacle set with enough steps for leg motion.
- Rerun low-small regression and require:
  - `fk_semantic_collision_count == 0`
  - `fk_semantic_collision_rate == 0`
  - covered crossing rows `> 0`
  - `planned_vs_fk_foot_error_crossing_leg_max_m <= 0.08m`

## Git Refs

- Baseline Ref: local dirty worktree after T302o timebase and lateral-bias diagnostics.
- Candidate Ref: local T302p implementation dirty worktree.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
  - `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  - `Go2Pvcnn/scripts/mpc_policy_eval.py`
  - `Go2Pvcnn/tests/test_batch_mpc_parametric.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - `Go2Pvcnn/tests/test_viewer_reset.py`
  - `Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py`
