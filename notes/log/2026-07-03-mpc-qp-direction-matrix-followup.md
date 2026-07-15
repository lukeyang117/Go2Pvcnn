# 2026-07-03 MPC QP Direction Matrix Follow-Up

## Purpose

Continue T302v against the full design requirement, not just the initial forward-lane acceptance. Broader low-small crossing must cover non-forward commands without semantic contact.

## Stage

MPC-QP backend / low-small crossing / viewer direction matrix.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Candidate Ref

Current worktree after adding low-small swing-over repair and knee-aware FK clearance lift in `mpc_qp`.

## Key Files

- `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
- `Go2Pvcnn/extension/batch_mpc_qp_planner/qp.py`
- `Go2Pvcnn/tests/test_mpc_qp_backend.py`
- `Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py`

## TDD Evidence

RED:

- `test_mpc_qp_lateral_command_low_small_crossing_keeps_fk_leg_over_obstacle` failed with missing `qp_low_small_swing_over_repair_count`.

GREEN:

- Added fixed-shape small-obstacle swing-over repair around nearby `semantic==1` candidates.
- Added `qp_low_small_swing_over_repair_count`.
- Extended FK clearance lift to consider knee semantic hits as well as shank hits.

## Commands And Results

Focused/static:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
```

Result: `14 passed`.

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `171 passed, 1 warning`.

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py

git diff --check
```

Result: both exit `0`.

Real GPU1 left regression:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'left:0.0,0.35,0.0' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m -0.12
```

Result: exit `0`, `viewer_crossing_acceptance_passed=true`, FK semantic collision `0`, touchdown-on-small `0`, small penetration `0`.

Real GPU1 8-command matrix, `lateral=-0.12`:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0;backward:-0.45,0.0,0.0;left:0.0,0.35,0.0;right:0.0,-0.35,0.0;diag_fl:0.32,0.32,0.0;diag_fr:0.32,-0.32,0.0;mixed_turn_l:0.45,0.15,0.6;mixed_turn_r:0.45,-0.15,-0.6' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m -0.12
```

Result: exit `0` process, summary acceptance `false`; `7/8` foot-over successes, FK semantic collision `0`, touchdown-on-small `0`, small penetration `0`; remaining miss is `diag_fl` lacking touchdown-after in this lane.

Real GPU1 8-command matrix, `lateral=+0.12`:

```bash
CUDA_VISIBLE_DEVICES=1 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_viewer_crossing_probe.py --device cuda:0 --commands 'forward:0.45,0.0,0.0;backward:-0.45,0.0,0.0;left:0.0,0.35,0.0;right:0.0,-0.35,0.0;diag_fl:0.32,0.32,0.0;diag_fr:0.32,-0.32,0.0;mixed_turn_l:0.45,0.15,0.6;mixed_turn_r:0.45,-0.15,-0.6' --cycles 1 --requested-n-frames 50 --playback-frames 50 --qp-iterations 1 --longitudinal-offset-m -0.35 --lateral-offset-m 0.12
```

Result: exit `0` process, summary acceptance `false`; `diag_fl` succeeds in this lane, but `mixed_turn_l` has `fk_semantic_collision_count=7` from knee semantic hits.

## Conclusion

The low-small swing-over repair improves non-forward coverage and fixes the previously failing `left` controlled case on GPU1. The full design acceptance is still not proven: `mixed_turn_l` with `lateral=+0.12` remains a hard body-leg semantic collision case.

## Follow-Up

Implement a true knee/body-leg avoidance repair or constraint, then rerun:

- both `lateral=-0.12` and `lateral=+0.12` 8-command matrices
- focused QP/current-MPC regression
- 1024/1024 GPU1 performance smoke, because the new repair adds fixed candidate sampling
