# T302k Parametric Trot-Phase Foot Curves

## Purpose

Fix the parametric MPC foot trajectory gait issue where all four feet moved together across the full horizon instead of alternating diagonal trot pairs.

## Stage

`extension/batch_mpc_planner` parametric MPC.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Code Changes

- Added a per-leg Bezier helper in `parametric.py` that accepts local leg phase instead of one global horizon phase.
- Moved `swing_center`, `swing_width`, `swing_prob`, and `contact_prob` earlier in decode so foot XY/Z curves can use the same gait timing.
- Foot XY and Z now advance only within each leg's swing window:
  - diagonal pair `[1, 2]` swings in the earlier default window;
  - diagonal pair `[0, 3]` swings in the later default window;
  - outside the active swing window, the leg holds its current foot position or touchdown endpoint.
- Added a regression test in `Go2Pvcnn/tests/test_batch_mpc_parametric.py` that first reproduced all four feet moving together and then passed after local phase gating.

## Verification

Local RED:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q -k trot_pairs
```

Result before implementation:

- failed because both diagonal pairs had frame-to-frame XY motion at the same frames.

Local GREEN and focused suite:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q -k trot_pairs
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

Results:

- trot-pair regression: `1 passed`.
- focused local suite: `225 passed, 1 warning`.
- warning is the existing tensor-to-float warning in `test_mpc_semantic_contact_avoidance_loss_has_xy_gradient_from_soft_field`.

IsaacLab GPU3 low-small smoke:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants parametric_v1 --requested-n-frames 300 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,pure_yaw:0.00 0.00 1.00' > tmp/t302k-parametric-mpc/low_small_parametric_v1_t302k_trotphase_gpu3.jsonl 2>&1
```

## Key Metrics

- `fk_foot_over_low_small_success_count=3/3`.
- `max_fk_stance_on_small_rate=0.0`.
- `max_fk_touchdown_on_small_rate=0.0`.
- `max_fk_foot_small_penetration_rate=0.0`.
- `max_terminal_planned_vs_fk_foot_error=1.923e-6m`.
- `max_touchdown_ik_fk_error=0.661m` remains open.

## Result

Pass for the immediate gait issue:

- The parametric foot curves no longer move all four feet together.
- Low-small IsaacLab smoke remains accepted after the trot-phase change.

## Follow-Up

- High-small/large rolling acceptance is still open and should be addressed with root nominal acceleration/bias and semantic/touchdown constraints.
- Touchdown endpoint quality remains open.

## Git Refs

- Baseline Ref: `1b799cd`
- Candidate Ref: working tree on top of `1b799cd`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
