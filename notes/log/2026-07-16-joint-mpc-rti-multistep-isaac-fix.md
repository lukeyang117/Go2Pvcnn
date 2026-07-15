# Joint MPC RTI Multi-step Isaac Fix

- Purpose: isolate and fix the real IsaacLab second-step process exit.
- Stage: `joint_mpc_rti` field synchronization and CUDA Graph runtime.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `643a172`; Candidate Ref: `joint_mpc` working tree.
- Root causes: the RayCaster observer launched the full PyTorch SDF build inside the Warp/Isaac sensor callback; the CUDA Graph runner returned captured output before its first replay materialized that output.
- Fix: RayCaster callbacks now only queue updated env ids, `latest_field()` consumes and publishes those rows outside the sensor callback, and graph construction performs one replay before exposing `captured_result`.
- TDD: deferred-publication test failed before the field-sync change; first-result materialization test failed before the graph replay change.
- Real 1-env result: 3 steps pass, field version `2`, finite reference, `target_step=1`, `x0_error_max=0.0`, final refresh `19.8738 ms`.
- Real 16-env result: 3 steps pass, all 16 field rows ready at version `2`, finite reference, `target_step=1`, `x0_error_max=0.0`, final refresh `19.9588 ms`.
- Real 1024-env trace: `gym.make`, reset, and the first `env.step` complete; the process stalls/exits during the first explicit `manager.refresh_from_env`, so scene creation is not the remaining boundary.
- Pure field scaling: empty-semantic `151×151` build measured about `18.7 ms` at B16, `47.9 ms` at B64, `277.0 ms` at B256, and `1136.3 ms` at B1024 with `2.94 GiB` peak allocation. The current tensor Jump Flood implementation is the real 1024 full-chain bottleneck.
- Regression: joint suite `72 passed`; old MPC subset `193 passed`.
- Performance context: accepted uncontended run remains `2885.63 ms/1000`; fresh runs during heavy shared-GPU contention measured `3454.87-4703.18 ms`, so no new uncontended acceptance number is claimed.
- Result: multi-step Isaac stability fixed through 16 environments; real 1024 full-chain timing remains open pending a Triton/CUDA JFA or static world-SDF cache.
