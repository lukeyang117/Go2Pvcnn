# T302k Parametric Current-Foot Replan Anchor

## Purpose

Record the local contract change requested after the trot-phase fix: every parametric MPC replan should start from the current IsaacLab-returned foot positions, rather than letting FK of the initial joint anchor rewrite frame0 feet.

## Stage

`extension/batch_mpc_planner` parametric MPC.

## Related Todo

[../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Command

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_batch_mpc_parametric.py \
  Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py \
  Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

## Input Conditions

- Working tree on top of T302k parametric MPC changes.
- `MpcRuntimeCfg.use_parametric_trajectory=True` default path.
- New red test perturbed current `state.foot_pos` and required `plan_segment(...).foot_pos[:, 0]` to match it.

## Key Metrics

- Red backend failure before fix: frame0 planned/exported foot mismatch max `0.125999987m`.
- Focused suite after fix: `227 passed, 1 warning`.

## Result

Pass locally.

`decode_parametric_trajectory()` already starts the target foot curve from current `state.foot_pos`. The mismatch was in `plan_segment()` output: after setting frame0 joints to current `state.joint_angles`, FK could rewrite frame0 `foot_pos`. The parametric path now writes `result.foot_pos[:, 0] = state.foot_pos` after FK export.

## Contract

- Frame0 root, joint, and foot outputs are current IsaacLab state anchors.
- Frame1+ `foot_pos` remains FK-realized from the clamped IK joint sequence.
- FK export tests now check frame1+ rather than requiring frame0 FK equality.

## Conclusion

This should reduce visible replan discontinuity and make the trajectory begin at the same foot positions IsaacLab just returned. It does not yet solve stance-root coupling over the future frames or high/large obstacle acceptance.

## Follow-Up

Run IsaacLab low-small/yaw smoke again before claiming visual behavior is fixed. Continue high/large work with root nominal acceleration/bias after current-foot anchoring is verified in simulation.

## Git Refs

- Baseline Ref: `working tree @ 1b799cd` plus T302k local changes before current-foot anchor.
- Candidate Ref: `working tree 2026-05-26 17:13 +0800`.
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
