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
- Real 1024-env attempt: no crash or OOM, GPU0 used about `6.9 GiB`, but first-time Isaac/compile initialization did not emit final JSON within 10 minutes and was stopped; steady-state real 1024 timing remains unverified.
- Regression: joint suite `72 passed`; old MPC subset `193 passed`.
- Performance context: accepted uncontended run remains `2885.63 ms/1000`; fresh runs during heavy shared-GPU contention measured `3454.87-4703.18 ms`, so no new uncontended acceptance number is claimed.
- Result: multi-step Isaac stability fixed through 16 environments; real 1024 initialization/timing remains open.

