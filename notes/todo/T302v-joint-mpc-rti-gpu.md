# T302v Joint MPC RTI GPU

## Current State

- Branch: `joint_mpc`; baseline commit: `cb2fff4`.
- Rolling contract: inject measured `x0`, optimize `H=16`, publish only `x1` to PPO reference rewards.
- Production profile: compiled fixed-horizon rollout/objective/query, packed geometry queries, diagonal GGN Riccati, `(1.0, 0.25)` line search, CUDA Graph replay.
- Full synchronous performance: `1024 × H16 × 1000` exact field + MPC refreshes = `4469.05 ms`, mean `4.469 ms`, P95 `4.492 ms`, max `4.655 ms`, field version `+1000`, nonfinite `0`, peak `858.99 MiB`.
- Verification: joint `93 passed`; old MPC/reward/viewer `193 passed`; public factory batches `1/40/512/1024` are finite and version-correct.
- Real IsaacLab: 1-env and 16-env three-step probes pass with field version `2`, finite `x1`, `target_step=1`, and `x0_error_max=0`; final refresh is about `19.9 ms`.

## Open Children

- Real IsaacLab 1024-env physics + RayCaster-ray timing remains a separate end-to-end boundary; planner acceptance now includes scanner buffers, exact field publication, RTI and x1, but excludes physics/raycast generation itself.

## Closed Children Archive

- Public contracts, kinematics, gait, terrain/SDF cache, continuous loss model, RTI solver, rolling manager, reward/viewer wiring.
- MPX reference reading: MPX is JAX multiple-shooting SQP with `jit(vmap)`, temporal `associative_scan`, parallel line search, and shifted warm starts; it is not MPPI/CEM.
- Packed objective/linearization queries, fixed-shape compilation, rollout reuse, and production CUDA Graph runner.
- T302v.1 multi-step stability: defer field construction out of the RayCaster callback and replay a newly captured graph once before returning its first result.
- T302v.2 synchronous exact EDT: tensor Jump Flood replaced by fixed-workspace CUDA warp-level exact EDT; query-time analytic gradients and repeated-row gathers remove full gradient/candidate-map copies.
- Batch-size contract: `create_trajectory_manager(..., num_envs=N)` is the upper-level entry; attach auto-forwards env count. Changing batch size requires rebuilding manager/cache/CUDA Graph and is rejected explicitly on an existing instance.

## Related Logs

- [Performance acceptance](../log/2026-07-15-joint-mpc-rti-performance.md)
- [Regression verification](../log/2026-07-15-joint-mpc-rti-regression.md)
- [IsaacLab smoke](../log/2026-07-15-joint-mpc-rti-isaac-smoke.md)
- [MPX reference mapping](../log/2026-07-15-mpx-reference-reading.md)
- [Multi-step Isaac fix](../log/2026-07-16-joint-mpc-rti-multistep-isaac-fix.md)

## Git Refs

- Last Feature Commit: `4bad2e0`
- Last Verified Commit: `4bad2e0` (`93` joint, `193` old MPC, full refresh acceptance)
- Current Work Ref: `joint_mpc`
- Key Files: `planner.py`, `runtime/cuda_graph.py`, `runtime/manager.py`, `solver/primal_dual_ilqr.py`, `joint_mpc_rti_perf_probe.py`.

## Next Step

Record the remaining real IsaacLab 1024-env physics/raycast boundary separately; do not reopen the planner field+MPC performance leaf unless fresh uncontended acceptance regresses.

## Node Details

The performance profile keeps the continuous terrain, semantic, full-body, posture, command, smoothness, and terminal losses. It adds no semantic hard gate, fixed avoidance direction, specified crossing leg, snapping, projection, or repair. `(1.0,)` was rejected because down-step root lowering failed; `(1.0, 0.25)` passes the current behavior suite and the 3-second performance target.
