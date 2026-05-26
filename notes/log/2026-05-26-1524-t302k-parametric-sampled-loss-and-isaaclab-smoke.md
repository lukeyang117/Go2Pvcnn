# T302k Parametric Sampled Loss And IsaacLab Smoke

## Purpose

Continue T302k after FK-realized default output by adding sampled-frame parametric losses, enabling a real parametric probe variant, and running IsaacLab smoke tests in `env_isaacsim`.

## Stage

`extension/batch_mpc_planner` parametric MPC.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Code Changes

- Added sampled parametric losses:
  - `parametric_reachability`
  - `parametric_terrain_clearance`
  - `parametric_semantic_contact`
  - `parametric_low_small_crossing`
  - `parametric_gait_regularization`
  - `parametric_command_progress`
  - `parametric_curve_regularization`
- Added an Adam optimization loop over `MpcParametricVariables` for `runtime.optimize_steps > 0`.
- Kept FK-realized export after optimization.
- Added `parametric_v1` probe variant for low-small and semantic obstacle probes.

## Verification

Local commands:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k parametric_optimization_reduces_low_small_crossing_cost
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py -q -k parametric
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py -q
```

Local results:

- Parametric optimization red/green: `1 passed`.
- Probe variant test: `1 passed, 34 deselected`.
- Parametric + low-small probe local: `43 passed`.
- Backend + parametric + low-small probe: `159 passed, 1 warning`.

IsaacLab commands used `env_isaacsim`:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants parametric_v1 --requested-n-frames 300 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,pure_yaw:0.00 0.00 1.00' > tmp/t302k-parametric-mpc/low_small_parametric_v1_gpu3.jsonl 2>&1
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --variants parametric_v1 --cases small,large --semantic-small-height-m 0.46 --requested-n-frames 300 --warmup-steps 6 > tmp/t302k-parametric-mpc/high_large_parametric_v1_gpu3.jsonl 2>&1
```

An initial GPU0 run failed due PhysX/CUDA OOM; GPU3 had enough free memory and completed.

## Key Metrics

Low-small GPU3:

- `forward_v050`: `fk_foot_over_low_small_success=1`, contact/penetration `0`, terminal FK error `~1.9e-6m`.
- `forward_yaw_v050_vy025_yaw100`: `fk_foot_over_low_small_success=1`, contact/penetration `0`, terminal FK error `~1.9e-6m`.
- `pure_yaw`: foot-over not required; contact/penetration `0`.
- Remaining endpoint issue: touchdown behind swing foot still about `0.48-0.62m`; mixed-yaw FK swing above root `0.134m`.

High-small/large GPU3:

- `semantic_task_violation_count=6/6`.
- `large_avoid_success_count=0`.
- `max_semantic_penetration_rate=0.0225`.
- `max_stance_on_semantic_rate=0.0459`.
- `max_touchdown_on_semantic_rate=0.0417`.

## Result

Partial:

- Parametric sampled losses and optimization run locally.
- IsaacLab can execute the parametric planner in `env_isaacsim`.
- FK-realized output remains aligned with playback.
- Low-small translation foot-over is now achieved.
- High-small/large semantic avoidance and endpoint/touchdown quality are not accepted yet.

## Follow-Up

Create/continue T302k.9:

- Add root path semantic avoidance for high-small/large.
- Add touchdown endpoint consistency so touchdown targets are not behind realized swing.
- Add mixed-yaw foot height guard to avoid swing feet rising above root.
- Re-run both GPU3 IsaacLab probes after T302k.9.

## Git Refs

- Baseline Ref: `1b799cd`
- Candidate Ref: working tree on top of `1b799cd`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
