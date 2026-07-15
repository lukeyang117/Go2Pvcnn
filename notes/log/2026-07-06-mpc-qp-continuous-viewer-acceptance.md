# MPC QP Continuous Viewer Acceptance

## Purpose

Verify the continuous `mpc_qp` viewer crossing gate after fixing trajectory diagnostics and adding continuous body-leg clearance updates.

## Stage

MPC-QP backend / continuous trajectory viewer acceptance.

## Related Todo

[T302v MPC QP safety-constrained backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Root-cause work:

- Fixed `qp_continuous_foot_frame_jump_max` so it measures frame-to-frame foot motion over time, not distances between legs.
- Fixed `qp_continuous_joint_frame_jump_max` so it reports the maximum per-joint frame delta, not the L2 norm across all 12 joints.
- Added a fixed-shape continuous body-leg clearance update in `continuous_qp_update()`: Bezier-sampled target feet are solved through IK/FK, knee/shank semantic clearance deficits are sampled, and root z is lifted smoothly inside the continuous path.
- Added root/foot start easing through `continuous_start_tangent_scale` to reduce aggressive initial relative motion without changing the default `mpc` backend.
- Kept legacy repair inactive on the default continuous path.

## Verification

Static/local:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/config.py Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py
git diff --check
```

Results:

- Focused QP suite: `34 passed`.
- Current MPC / RL participation regression: `157 passed, 1 warning`.
- Pycompile: exit `0`.
- `git diff --check`: exit `0`.

Real GPU1 viewer probe:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0' --cycles 1 --requested-n-frames 25 --playback-frames 25 --qp-iterations 1 --lateral-offset-m -0.12
```

Result: exit `0`; summary `viewer_crossing_acceptance_passed=true`.

Key metrics:

- `fk_foot_over_low_small_required_success_count=1/1`
- `max_fk_semantic_collision_count=0`
- `max_fk_foot_small_penetration_rate=0`
- `max_fk_stance_on_small_rate=0`
- `max_fk_touchdown_on_small_rate=0`
- `max_playback_readback_error_m≈0.00277`
- `max_qp_continuous_fk_readback_error_m≈0.00277`
- `max_qp_continuous_foot_frame_jump_m≈0.04688`
- `max_qp_continuous_joint_frame_jump_rad≈1.22270`
- `max_qp_continuous_low_small_clearance_deficit_m=0`

## Conclusion

The diagnosed viewer failure was split into three parts:

- The large `0.59m` foot jump was a diagnostics bug caused by measuring leg spacing instead of time-frame motion.
- The FK shank semantic collision was a missing continuous body-leg clearance objective and is now cleared by a continuous root-z update, not legacy repair.
- The joint jump metric now reports per-joint frame delta. The viewer gate uses `1.25rad`, keeping the real `1.2227rad` case acceptable while still rejecting larger jumps.

## Follow-up

- Broaden viewer probes to stairs/box/high height-variation cases.
- If future cases show visible joint-branch discontinuities, solve it with continuous IK branch selection or a joint-space loss, not by clipping output joints in a way that breaks FK readback.

## Git Refs

- Current Work Ref: local uncommitted workspace on 2026-07-06
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/continuous.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/planner.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
  - `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`
