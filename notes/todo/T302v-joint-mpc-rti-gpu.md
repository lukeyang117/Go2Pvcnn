# T302v Joint MPC RTI GPU

## Current State

- Branch: `joint_mpc`; baseline commit: `cb2fff4`.
- Rolling contract: inject measured `x0`, optimize `H=16`, publish only `x1` to PPO reference rewards.
- Production profile: compiled fixed-horizon rollout/objective/query, packed geometry queries, diagonal GGN Riccati, `(1.0, 0.25)` line search, CUDA Graph replay.
- Performance: `1024 × H16 × 1000 = 2885.63 ms`, mean `2.886 ms`, nonfinite `0`, peak `282.58 MiB`.
- Verification: joint `71 passed`; old MPC/reward/viewer `193 passed`.
- Real IsaacLab: 1-env/1-step finite reference, `target_step=1`, field ready, `x0_error_max=0`; 2-step process still exits before final JSON.

## Open Children

- T302v.1: isolate the real IsaacLab second-step hard exit from scanner update, Isaac process state, and CUDA Graph runtime.
- T302v.2: run larger real IsaacLab steady-state timing after T302v.1.

## Closed Children Archive

- Public contracts, kinematics, gait, terrain/SDF cache, continuous loss model, RTI solver, rolling manager, reward/viewer wiring.
- MPX reference reading: MPX is JAX multiple-shooting SQP with `jit(vmap)`, temporal `associative_scan`, parallel line search, and shifted warm starts; it is not MPPI/CEM.
- Packed objective/linearization queries, fixed-shape compilation, rollout reuse, and production CUDA Graph runner.

## Related Logs

- [Performance acceptance](../log/2026-07-15-joint-mpc-rti-performance.md)
- [Regression verification](../log/2026-07-15-joint-mpc-rti-regression.md)
- [IsaacLab smoke](../log/2026-07-15-joint-mpc-rti-isaac-smoke.md)
- [MPX reference mapping](../log/2026-07-15-mpx-reference-reading.md)

## Git Refs

- Last Feature Commit: `cb2fff4` plus current working tree
- Last Verified Commit: `cb2fff4` plus current working tree
- Current Work Ref: `joint_mpc`
- Key Files: `planner.py`, `runtime/cuda_graph.py`, `runtime/manager.py`, `solver/primal_dual_ilqr.py`, `joint_mpc_rti_perf_probe.py`.

## Next Step

Capture the second-step Isaac process exit with a scanner-update-only vs graph-replay-only split probe; do not change planner behavior without evidence.

## Node Details

The performance profile keeps the continuous terrain, semantic, full-body, posture, command, smoothness, and terminal losses. It adds no semantic hard gate, fixed avoidance direction, specified crossing leg, snapping, projection, or repair. `(1.0,)` was rejected because down-step root lowering failed; `(1.0, 0.25)` passes the current behavior suite and the 3-second performance target.
