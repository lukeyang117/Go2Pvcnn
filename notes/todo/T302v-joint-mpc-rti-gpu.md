# T302v Joint MPC RTI GPU

## Current State

- Branch: `joint_mpc`; baseline commit: `cb2fff4`.
- Rolling contract: inject measured `x0`, optimize `H=16`, publish only `x1` to PPO reference rewards.
- Production profile: compiled fixed-horizon rollout/objective/query, packed geometry queries, diagonal GGN Riccati, `(1.0, 0.25)` line search, CUDA Graph replay.
- Last accepted unsigned-field synchronous performance: `1024 × H16 × 1000` exact field + MPC refreshes = `4469.05 ms`, mean `4.469 ms`, P95 `4.492 ms`, max `4.655 ms`, field version `+1000`, nonfinite `0`, peak `858.99 MiB`. The signed-field candidate has not inherited this acceptance.
- Current functional verification: joint MPC `110 passed`; old MPC/reward/viewer `213 passed`; public factory/cache batches `1/40/512/1024` are finite and version-correct.
- Real IsaacLab: 1-env and 16-env three-step probes pass with field version `2`, finite `x1`, `target_step=1`, and `x0_error_max=0`; final refresh is about `19.9 ms`.
- T302v.3 viewer foot-flying is fixed: shared name-based joint conversion gives zero adapter error; persistent stance anchors and physical `0.022m` foot contact offset keep the real nine-command stance surface residual at or below `0.010303m`; max joint step is `0.183284rad`; zero-command fixed trot remains active while root drift is numerically zero.
- T302v.4 functional implementation is verified: small/large semantic fields are signed and half-cell corrected; H16 covers stance-swing-stance; foot/calf/thigh continuous residuals participate in merit and GGN/LQ; native sphere/cuboid/cylinder/capsule/cone at `0.1/0.2/0.4m/s` cross `254/254`, with foot/calf/thigh/base collision frames and maximum penetration all zero, stance-on-small `0`, invalid `0`.
- Real nine-command viewer verification remains green on the current candidate: `passed=true`, joint-order error `0`, stance gap max `0.010098m`, joint step max `0.183805rad`, actual/planner foot error max `5.38e-7m`, and zero-command root drift remains numerical zero.

## Open Children

- Current signed-field candidate five-second performance: all four GPUs are occupied by external workloads. Do not accept or reject throughput from contested measurements; rerun `1024 x H16 x 1000` synchronous signed field + MPC on an idle card and require `<5s`, field version `+1000`, and nonfinite `0`.
- T302v.5 stop-on-small support viability: walking onto a native small object and switching to zero command leaves scheduled stance active but retracts all feet from the local surface. A `65`-case shape x stop-phase scan reproduces floating in every case and zero grounded support for at least `64` consecutive hold cycles. Flat and height-only controls largely ground; semantic-only and full semantic+height do not. Loss ablation identifies `small_object_foot_over` as the primary trigger. Any fix must preserve T302v.4 strict cross `254/254`, per-part collision frames `0`, stance-on-small `0`, flat grounding, and the no-hard-gate requirement.
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
- T302v.4 functional crossing: signed distance, thigh geometry/Jacobians, continuous GGN/LQ link clearance, effective single-height-map object top, touchdown avoidance, and native-shape acceptance are implemented without hard behavior gates.

## Related Logs

- [Performance acceptance](../log/2026-07-15-joint-mpc-rti-performance.md)
- [Regression verification](../log/2026-07-15-joint-mpc-rti-regression.md)
- [IsaacLab smoke](../log/2026-07-15-joint-mpc-rti-isaac-smoke.md)
- [MPX reference mapping](../log/2026-07-15-mpx-reference-reading.md)
- [Multi-step Isaac fix](../log/2026-07-16-joint-mpc-rti-multistep-isaac-fix.md)
- [Viewer foot-flying reproduction](../log/2026-07-16-joint-mpc-rti-viewer-foot-flying-reproduction.md)
- [Viewer grounding fix](../log/2026-07-16-joint-mpc-rti-viewer-grounding-fix.md)
- [Viewer foot-name fix](../log/2026-07-16-joint-mpc-viewer-foot-name-fix.md)
- [Small-obstacle collision quantification](../log/2026-07-16-joint-mpc-small-obstacle-collision-quantification.md)
- [Small-obstacle crossing design](../log/2026-07-16-joint-mpc-rti-small-obstacle-crossing-design.md)
- [Small-obstacle crossing implementation](../log/2026-07-16-joint-mpc-rti-small-obstacle-crossing-implementation.md)
- [Stop-on-small floating reproduction](../log/2026-07-16-joint-mpc-rti-stop-on-small-floating-reproduction.md)

## Git Refs

- Last Feature Commit: `625768f`.
- Last Verified Commit: `625768f` (`110` joint, `213` legacy, real nine-command viewer and crossing matrix pass; five-second signed-field performance open).
- Current Work Ref: `joint_mpc`
- Key Files: `planner.py`, `runtime/cuda_graph.py`, `runtime/manager.py`, `solver/primal_dual_ilqr.py`, `joint_mpc_rti_perf_probe.py`.

## Next Step

Design and implement T302v.5 support recovery without a hard zero-command/semantic gate, then reverify strict cross/zero collision/stance grounding. Keep the uncontended `1024 x H16 x 1000` signed-field performance gate open in parallel.

## Node Details

The performance profile keeps the continuous terrain, semantic, full-body, posture, command, smoothness, and terminal losses. It adds no semantic hard gate, fixed avoidance direction, specified crossing leg, snapping, projection, or repair. `(1.0,)` was rejected because down-step root lowering failed; `(1.0, 0.25)` passes the current behavior suite and the 3-second performance target.
