# T302v Joint MPC RTI GPU

## Current State

- Branch: `joint_mpc`; baseline commit: `cb2fff4`.
- Rolling contract: inject measured `x0`, optimize one fixed-shape full stance-swing-stance horizon, publish only `x1` to PPO reference rewards. H16 remains the baseline; Stage A may explore H16-H50 with `Horizon = 2 * half_cycle_steps` before selecting `H_selected`.
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
- T302v.7 Chinese HTML design is approved with a three-stage completion rule. It inherits all three prior Joint MPC designs, specifies complete root-joint FK/GGN cross blocks, scheduled stance equality, horizon command progress, command-conditioned touchdown, 20-80ms foot-leading-root startup, arrowhead/Schur Riccati, and one scenario-metric JointMetrics contract. Stage A jointly explores H16-H50 fixed periods and the original solver/loss directions, then selects the shortest stable `H_selected` passing every behavior metric. Stage B freezes that exact horizon and uses MPX-referenced temporal/state-space GPU parallelism to reach `<=5.0s/1000` without relaxing the threshold for longer trajectories. Stage C reruns both complete gates on the same final candidate. See [design log](../log/2026-07-17-joint-mpc-root-joint-coupled-gait-design.md).
- The amended 14-task TDD implementation plan is written and inline execution is authorized. Tasks 1-9 close Stage A behavior and select `H_selected`, tasks 10-13 close the unchanged selected-horizon performance gate with MPX-referenced parallelism, and task 14 performs the mandatory same-candidate joint rerun. See [plan log](../log/2026-07-17-joint-mpc-root-joint-coupled-gait-plan.md).
- The current uncommitted Stage A continuation has a constrained-Riccati/active-joint-bound candidate with `88` focused tests passing, near-unity one-placement command tracking, and full crossing coverage, but the new joint gate is **not closed**: rare continuing-stance bound conflicts still produce centimeter x1 slip plus nonzero foot/calf collision or stance-on-small frames. Performance remains deferred. See [constrained RTI diagnostics](../log/2026-07-17-joint-mpc-stage-a-constrained-rti-diagnostics.md).

## Open Children

- T302v.7 support-driven gait quality: define and implement a contract for near-zero world-frame stance slip, command-conditioned swing touchdown lead, and root progress coupled to established support. Current grounding-Z/collision metrics do not detect root-carried feet.
- Realistic multi-cell signed-field performance: the final contract is `1024 x H_selected x 1000 <=5s` with Stage A's selected horizon and unchanged `11x11` small plus `41x41` large footprints. The current H16 baseline misses this gate; single-cell results must not be used as acceptance. Stage B starts only after Stage A and may use MPX-style temporal associative scans, multiple shooting, parallel line search and state-space Schur structure in addition to exact-EDT work.
- Real IsaacLab 1024-env physics + RayCaster-ray timing remains a separate end-to-end boundary; planner acceptance now includes scanner buffers, exact field publication, RTI and x1, but excludes physics/raycast generation itself.

## Closed Children Archive

- Public contracts, kinematics, gait, terrain/SDF cache, continuous loss model, RTI solver, rolling manager, reward/viewer wiring.
- MPX reference reading: MPX is JAX multiple-shooting SQP with `jit(vmap)`, temporal `associative_scan`, parallel line search, and shifted warm starts; it is not MPPI/CEM.
- 2026-07-17 Stage B pause: viewer/production CUDA Graph capture now passes after graph-safe `solve_ex/cholesky_ex`, cached CUDA constants, capture-safe validation and complete solver-state fixed-address copy-back. A fixed-size Triton SPD solve replaces MAGMA in the coupled Riccati, so `1/40/512/1024` batches capture with finite outputs instead of failing at `magma_queue::setup_ptrArray`. Busy-GPU 1024xH16 screening measured field `6.54-8.94ms`, MPC `61.30-88.47ms`, full `69.20-95.30ms`; these are diagnostic only and Stage B is not passed. Fixed general solve and conditional-factor associativity pass independently, but PyTorch 2.7 generic `associative_scan` fails inside its symbolic-vmap `matmul`; no incomplete associative solver is routed into planner. Stage B is paused by user direction while crossing roll/pitch and airborne-touchdown behavior is diagnosed.
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
- [Root-joint coupled gait implementation plan](../log/2026-07-17-joint-mpc-root-joint-coupled-gait-plan.md)
- [Stage A constrained RTI diagnostics](../log/2026-07-17-joint-mpc-stage-a-constrained-rti-diagnostics.md)
- [H30 adaptive contact and root assist design](../log/2026-07-18-joint-mpc-h30-adaptive-contact-root-assist-design.md)
- [H30 implementation plan amendment](../log/2026-07-18-joint-mpc-h30-implementation-plan-amendment.md)

## Git Refs

- Last Feature Commit: `9e71ac1`.
- Last Verified Commit: `9e71ac1` (functional gates pass; realistic multi-cell signed performance blocked).
- Current Work Ref: `joint_mpc`
- Key Files: `planner.py`, `runtime/cuda_graph.py`, `runtime/manager.py`, `solver/primal_dual_ilqr.py`, `joint_mpc_rti_perf_probe.py`.

## Next Step

After user review, amend the inherited implementation plan around the fixed H30 contract: Stage A closes H30/15+15 adaptive contact, bounded root assistance and every old/new JointMetrics field; Stage B then closes realistic `1024 x H30 x 1000 <=5.0s` using the recorded MPX-referenced GPU work; Stage C reruns both complete gates on the same final candidate.

## Node Details

The performance profile keeps the continuous terrain, semantic, full-body, posture, command, smoothness, and terminal losses. It adds no semantic hard gate, fixed avoidance direction, specified crossing leg, snapping, projection, or repair. `(1.0,)` was rejected because down-step root lowering failed; `(1.0, 0.25)` passes the current behavior suite and the 3-second performance target.
