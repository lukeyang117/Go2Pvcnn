# T302v Joint MPC RTI GPU

## Current State

- Branch: `joint_mpc`; baseline commit: `cb2fff4`.
- Rolling contract: inject measured `x0`, optimize `H=16`, publish only `x1` to PPO reference rewards.
- Production profile: compiled fixed-horizon rollout/objective/query, packed geometry queries, diagonal GGN Riccati, `(1.0, 0.25)` line search, CUDA Graph replay.
- Performance: `1024 × H16 × 1000 = 2885.63 ms`, mean `2.886 ms`, nonfinite `0`, peak `282.58 MiB`.
- Verification: joint `72 passed`; old MPC/reward/viewer `193 passed`.
- Real IsaacLab: 1-env and 16-env three-step probes pass with field version `2`, finite `x1`, `target_step=1`, and `x0_error_max=0`; final refresh is about `19.9 ms`.

## Open Children

- T302v.2: replace the current tensor Jump Flood field build (`B1024≈1136ms`, peak `2.94GiB`) with a fixed Triton/CUDA JFA or static world-SDF cache, then rerun real 1024-env steady-state timing.

## Closed Children Archive

- Public contracts, kinematics, gait, terrain/SDF cache, continuous loss model, RTI solver, rolling manager, reward/viewer wiring.
- MPX reference reading: MPX is JAX multiple-shooting SQP with `jit(vmap)`, temporal `associative_scan`, parallel line search, and shifted warm starts; it is not MPPI/CEM.
- Packed objective/linearization queries, fixed-shape compilation, rollout reuse, and production CUDA Graph runner.
- T302v.1 multi-step stability: defer field construction out of the RayCaster callback and replay a newly captured graph once before returning its first result.

## Related Logs

- [Performance acceptance](../log/2026-07-15-joint-mpc-rti-performance.md)
- [Regression verification](../log/2026-07-15-joint-mpc-rti-regression.md)
- [IsaacLab smoke](../log/2026-07-15-joint-mpc-rti-isaac-smoke.md)
- [MPX reference mapping](../log/2026-07-15-mpx-reference-reading.md)
- [Multi-step Isaac fix](../log/2026-07-16-joint-mpc-rti-multistep-isaac-fix.md)

## Git Refs

- Last Feature Commit: `643a172` plus current working tree
- Last Verified Commit: `643a172` plus current working tree
- Current Work Ref: `joint_mpc`
- Key Files: `planner.py`, `runtime/cuda_graph.py`, `runtime/manager.py`, `solver/primal_dual_ilqr.py`, `joint_mpc_rti_perf_probe.py`.

## Next Step

Implement and benchmark a fixed Triton/CUDA JFA or static world-SDF cache; the real trace already shows scene startup completes and the first refresh is blocked by field construction.

## Node Details

The performance profile keeps the continuous terrain, semantic, full-body, posture, command, smoothness, and terminal losses. It adds no semantic hard gate, fixed avoidance direction, specified crossing leg, snapping, projection, or repair. `(1.0,)` was rejected because down-step root lowering failed; `(1.0, 0.25)` passes the current behavior suite and the 3-second performance target.
