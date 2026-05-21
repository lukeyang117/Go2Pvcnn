# 2026-05-21 14:20 MPC Grounded Touchdown Output Lock

## Purpose

Record the small MPC decode change requested during T302g discussion: decoded touchdowns must land on the height map, touchdown-following stance frames must keep the same foot coordinate, and frame 0 must remain anchored to the current Go2 state for viewer handoff.

## Stage

`extension/batch_mpc_planner` decode path.

## Related Todo

- [T302g MPC Semantic RL Training Config](../todo/T302g-mpc-semantic-rl-training-config.md)
- [T302 MPC Body/Leg Collision Safety](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Command / Procedure

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/variables.py Go2Pvcnn/extension/batch_mpc_planner/optimizer.py Go2Pvcnn/tests/test_batch_mpc_backend.py
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "frame_zero_state_anchor or frame_zero_joint_state_anchor or decode_grounded_touchdowns or grounded_touchdowns"
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

## Input Conditions

- Code path: `plan_segment(...)` with `planner_backend="mpc"`.
- Scope: `decode_trajectory(..., terrain=terrain)` and optimizer decode calls.
- Semantic obstacle exclusion is not included in this slice.

## Key Metrics

- Targeted backend tests: `4 passed, 92 deselected`.
- Full backend tests: `96 passed, 1 warning`.
- `py_compile`: exit `0`.

## Result

Pass.

## Conclusion

`decode_trajectory(..., terrain=terrain)` now computes touchdown `xy` from the current sampled foot trajectory, samples touchdown `z` from `height_at(terrain, xy)`, and locks decoded foot positions from the touchdown frame through the remaining stance horizon to that grounded point. Optimizer loop decodes pass `terrain`, so losses and final export now see the same grounded/locked foot trajectory. Frame-0 root/rpy/foot residuals are gated to zero, touchdown locking skips frame 0, and `plan_segment(...)` writes frame-0 joint angles from the input state so viewer replans start from the current displayed Go2 state.

## Follow-Up

- Real viewer/runtime inspection still needs to confirm visual touchdown markers no longer appear airborne.
- This slice does not yet add semantic no-small-obstacle touchdown selection.

## Git Refs

- Baseline Ref: working tree before 2026-05-21 14:20 CST change.
- Candidate Ref: working tree after grounded touchdown postprocess change.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py](../../Go2Pvcnn/extension/batch_mpc_planner/optimizer.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
