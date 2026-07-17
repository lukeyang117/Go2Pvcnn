# T302v Joint MPC RTI GPU

## Current State

- Branch: `joint_mpc`; baseline commit: `cb2fff4`.
- Rolling contract: inject measured `x0`, optimize `H=16`, publish only `x1` to PPO reference rewards.
- Production profile: compiled fixed-horizon rollout/objective/query, packed geometry queries, diagonal GGN Riccati, `(1.0, 0.25)` line search, CUDA Graph replay.
- Signed-field performance remains open after strengthening the probe from single cells to realistic multi-cell footprints. The earlier `4768.28ms` result is rejected because it hit a single-cell fast path; `11x11` small plus `41x41` large maps remain above the `5s` full-refresh gate even after MPC compilation.
- Current functional verification: joint MPC `119 passed`; old MPC/reward/viewer `213 passed`; public factory/cache batches `1/40/512/1024` are finite and version-correct.
- Real IsaacLab: 1-env and 16-env three-step probes pass with field version `2`, finite `x1`, `target_step=1`, and `x0_error_max=0`; final refresh is about `19.9 ms`.
- T302v.3 viewer foot-flying is fixed: shared name-based joint conversion gives zero adapter error; persistent stance anchors and physical `0.022m` foot contact offset keep the real nine-command stance surface residual at or below `0.010303m`; max joint step is `0.183284rad`; zero-command fixed trot remains active while root drift is numerically zero.
- T302v.4 functional implementation is verified: small/large semantic fields are signed and half-cell corrected; H16 covers stance-swing-stance; foot/calf/thigh continuous residuals participate in merit and GGN/LQ; native sphere/cuboid/cylinder/capsule/cone at `0.1/0.2/0.4m/s` cross `254/254`, with foot/calf/thigh/base collision frames and maximum penetration all zero, stance-on-small `0`, invalid `0`.
- T302v.5 stop-on-small support recovery is implemented and verified without hard command/semantic gates: RTI root warm starts re-base to the current command, foot-over is mid-swing shaped, safe landing/support use internally matched continuous margins, and native collision probes use queried local shape height. The final 65-case stop matrix is `65/65`, maximum consecutive zero-support `1`, root drift `0`, stance-on-small `0`, and all per-part collision frames `0`; the 160-step crossing matrix is `100%` overall/per case with the same zero-collision result.
- Real nine-command viewer verification remains green on the current candidate: `passed=true`, joint-order error `0`, stance gap max `0.0114493m`, joint step max `0.185368rad`, actual/planner foot error max `5.24e-7m`, and zero-command root drift remains numerical zero.
- T302v performance continuation compiles the fixed-shape LQ/query/rollout path, reducing MPC to about `2.55-2.85ms` without changing losses or geometry. Exact signed EDT remains the blocker on realistic multi-cell maps. See [full design revalidation](../log/2026-07-17-joint-mpc-rti-full-design-revalidation.md).
- T302v.7 root/foot propulsion-order diagnostics confirm the viewer observation: across eight flat rolling commands, consecutive-stance feet move by `1.040x` the signed root step on average, only `4.02%` stay within `1mm/frame`, and swing motion relative to root contributes only `7.0%` of root progress on average. The root is independently command-integrated while the phase-only swing target has no command-conditioned foothold. See [root/foot quantification](../log/2026-07-17-joint-mpc-root-foot-propulsion-order-quantification.md).
- T302v.7 Chinese HTML design is written for review. It inherits all three prior Joint MPC designs, specifies complete root-joint FK/GGN cross blocks, scheduled stance equality, horizon command progress, command-conditioned touchdown, 20-80ms foot-leading-root startup, arrowhead/Schur Riccati, and one scenario-metric JointMetrics contract. Work is sequenced: all walking/safety metrics first, then freeze behavior and reduce the idle-GPU realistic baseline from `7.4025s/1000` to `<=5.0s/1000`. See [design log](../log/2026-07-17-joint-mpc-root-joint-coupled-gait-design.md).

## Open Children

- T302v.7 support-driven gait quality: define and implement a contract for near-zero world-frame stance slip, command-conditioned swing touchdown lead, and root progress coupled to established support. Current grounding-Z/collision metrics do not detect root-carried feet.
- Realistic multi-cell signed-field performance: `1024 x H16 x 1000 <=5s` is not met with `11x11` small and `41x41` large footprints. Single-cell results must not be used as acceptance. Local exact-EDT variants were exhausted; next progress requires a new batched exact EDT architecture or an explicit contract change.
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
- T302v.5 stop-on-small support recovery: 65 native shape/offset cases pass with maximum one consecutive zero-support frame, no root drift, no stance-on-small, and no foot/calf/thigh/base collisions; RTI root command re-basing and continuous phase/signed-distance losses preserve crossing and real viewer behavior.
- T302v.6 performance investigation: compiled fixed-shape LQ/query/rollout is retained; single-cell EDT fusion, complementary transforms, stream chunking, brute-force reduction, and compact warp bbox experiments were evaluated. The single-cell pass was rejected and the failed CUDA experiments were removed.

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
- [Stop-on-small support recovery](../log/2026-07-16-joint-mpc-rti-stop-on-small-support-recovery.md)
- [Full design revalidation and signed performance closure](../log/2026-07-17-joint-mpc-rti-full-design-revalidation.md)
- [Root/foot propulsion-order quantification](../log/2026-07-17-joint-mpc-root-foot-propulsion-order-quantification.md)
- [Root-joint coupled gait design](../log/2026-07-17-joint-mpc-root-joint-coupled-gait-design.md)

## Git Refs

- Last Feature Commit: `9e71ac1`.
- Last Verified Commit: `9e71ac1` (functional gates pass; realistic multi-cell signed performance blocked).
- Current Work Ref: `joint_mpc`
- Key Files: `planner.py`, `runtime/cuda_graph.py`, `runtime/manager.py`, `solver/primal_dual_ilqr.py`, `joint_mpc_rti_perf_probe.py`.

## Next Step

User reviews the T302v.7 HTML design. After approval, write the implementation plan: Stage A closes all old/new JointMetrics; Stage B freezes behavior and closes the realistic `7.4025s -> <=5.0s` performance gate.

## Node Details

The performance profile keeps the continuous terrain, semantic, full-body, posture, command, smoothness, and terminal losses. It adds no semantic hard gate, fixed avoidance direction, specified crossing leg, snapping, projection, or repair. `(1.0,)` was rejected because down-step root lowering failed; `(1.0, 0.25)` passes the current behavior suite and the 3-second performance target.
