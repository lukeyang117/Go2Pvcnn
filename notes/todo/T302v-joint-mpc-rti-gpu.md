# T302v Joint MPC RTI GPU

## Current State

- Branch: `joint_mpc`; baseline commit: `cb2fff4`.
- Rolling contract: inject measured `x0`, optimize `H=16`, publish only `x1` to PPO reference rewards.
- Production profile: compiled fixed-horizon rollout/objective/query, packed geometry queries, diagonal GGN Riccati, `(1.0, 0.25)` line search, CUDA Graph replay.
- Full synchronous performance: `1024 × H16 × 1000` exact field + MPC refreshes = `4469.05 ms`, mean `4.469 ms`, P95 `4.492 ms`, max `4.655 ms`, field version `+1000`, nonfinite `0`, peak `858.99 MiB`.
- Verification: joint plus viewer `133 passed`; old MPC/reward/viewer `193 passed`; public factory batches `1/40/512/1024` are finite and version-correct.
- Real IsaacLab: 1-env and 16-env three-step probes pass with field version `2`, finite `x1`, `target_step=1`, and `x0_error_max=0`; final refresh is about `19.9 ms`.
- T302v.3 viewer foot-flying is fixed: shared name-based joint conversion gives zero adapter error; persistent stance anchors and physical `0.022m` foot contact offset keep the real nine-command stance surface residual at or below `0.010303m`; max joint step is `0.183284rad`; zero-command fixed trot remains active while root drift is numerically zero.

## Open Children

- Current candidate five-second performance recheck: all four GPUs were occupied by external training; contested `1024 x H16 x 1000` samples varied from `5.63s` to `8.99s`, so the prior uncontended `4.469s` acceptance is retained but not transferred to the new candidate without an idle-card rerun.
- Real IsaacLab 1024-env physics + RayCaster-ray timing remains a separate end-to-end boundary; planner acceptance now includes scanner buffers, exact field publication, RTI and x1, but excludes physics/raycast generation itself.

## Closed Children Archive

- Public contracts, kinematics, gait, terrain/SDF cache, continuous loss model, RTI solver, rolling manager, reward/viewer wiring.
- MPX reference reading: MPX is JAX multiple-shooting SQP with `jit(vmap)`, temporal `associative_scan`, parallel line search, and shifted warm starts; it is not MPPI/CEM.
- Packed objective/linearization queries, fixed-shape compilation, rollout reuse, and production CUDA Graph runner.
- T302v.1 multi-step stability: defer field construction out of the RayCaster callback and replay a newly captured graph once before returning its first result.
- T302v.2 synchronous exact EDT: tensor Jump Flood replaced by fixed-workspace CUDA warp-level exact EDT; query-time analytic gradients and repeated-row gathers remove full gradient/candidate-map copies.
- Batch-size contract: `create_trajectory_manager(..., num_envs=N)` is the upper-level entry; attach auto-forwards env count. Changing batch size requires rebuilding manager/cache/CUDA Graph and is rejected explicitly on an existing instance.
- T302v.3 joint order and stance grounding: fixed and real-verified across zero, forward/backward, lateral, yaw, speed-varied, and mixed commands without disabling fixed trot.
- Viewer actual-state foot ordering now shares the public articulation-name normalizer; the reported post-playback `NameError` is real-Isaac verified fixed.

## Related Logs

- [Performance acceptance](../log/2026-07-15-joint-mpc-rti-performance.md)
- [Regression verification](../log/2026-07-15-joint-mpc-rti-regression.md)
- [IsaacLab smoke](../log/2026-07-15-joint-mpc-rti-isaac-smoke.md)
- [MPX reference mapping](../log/2026-07-15-mpx-reference-reading.md)
- [Multi-step Isaac fix](../log/2026-07-16-joint-mpc-rti-multistep-isaac-fix.md)
- [Viewer foot-flying reproduction](../log/2026-07-16-joint-mpc-rti-viewer-foot-flying-reproduction.md)
- [Viewer grounding fix](../log/2026-07-16-joint-mpc-rti-viewer-grounding-fix.md)
- [Viewer foot-name fix](../log/2026-07-16-joint-mpc-viewer-foot-name-fix.md)

## Git Refs

- Last Feature Commit: `b99cda0`
- Last Verified Commit: `b99cda0` (`133` joint/viewer, real Isaac actual-state read pass; five-second perf recheck open)
- Current Work Ref: `joint_mpc`
- Key Files: `planner.py`, `runtime/cuda_graph.py`, `runtime/manager.py`, `solver/primal_dual_ilqr.py`, `joint_mpc_rti_perf_probe.py`.

## Next Step

Rerun `1024 x H16 x 1000` synchronous full refresh on an uncontended GPU before transferring the five-second performance acceptance to the current candidate. The real IsaacLab 1024-env physics/raycast boundary remains independent.

## Node Details

The performance profile keeps the continuous terrain, semantic, full-body, posture, command, smoothness, and terminal losses. It adds no semantic hard gate, fixed avoidance direction, specified crossing leg, snapping, projection, or repair. `(1.0,)` was rejected because down-step root lowering failed; `(1.0, 0.25)` passes the current behavior suite and the 3-second performance target.
