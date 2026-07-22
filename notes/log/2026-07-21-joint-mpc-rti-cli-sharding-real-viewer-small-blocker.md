# Joint MPC RTI CLI Sharding And Real Viewer Small Blocker

## Purpose

Verify Task 14 acceptance-runner sharding at the CLI boundary and implement the missing real-viewer small-obstacle actual-state path.

## Stage

- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan: Task 14 Steps 4-5
- Device: CPU for CLI preflight/tests; `cuda:1` for monitored Isaac viewer

## Changes

- Confirmed existing deterministic contiguous sharding and exact-key merge through real CLI report files.
- Added viewer `small` mode using a real S4 small obstacle and the real 0.01m semantic scanner field.
- Each cycle executes one complete H30 SQP/RTI, directly publishes x1, then reads actual root/joint/foot from Isaac.
- Reused `_small_detector_row`, `strict_crossing_event`, and `evaluate_trace(..., scenario="small")`; no private collision or crossing gate was added.
- Snapshot measured x0 before playback to avoid Isaac tensor-alias corruption of x0 diagnostics.
- Use a representative S4 sphere and stop one 24-cycle gait period after first strict crossing so later course obstacles do not contaminate the target event.

## Verification

CLI two-shard preflight:

- shard 0 and shard 1 each wrote one exact cell with `formal_complete=false`;
- reverse-order merge restored zero then forward key order exactly once;
- merged report had `merged_shards=2` and `formal_complete=true`;
- the two-step forward cell remained gate-red because it had not crossed, which is expected and proves merge preserves failures.

Focused regression: `46 passed`; viewer contract subset: `18 passed`; full package: `207 passed in 43.43s`; compileall and diff check passed.

Monitored real viewer representative sphere event:

- command: `(1.0, 0.0, 0.0)`;
- actual cycles: `49` (first strict crossing plus one 24-cycle gait tail);
- strict crossing: `1.0`;
- foot/knee/calf/thigh/base collision rates: all `0`;
- maximum penetration, touchdown-on-small, stance-on-small, airborne touchdown: all `0`;
- map-valid and trajectory-valid ratios: `1.0`;
- x0 injection error: `0`; published x1 error below `1e-6`;
- cold starts: `1`; unexpected cold restarts: `0`;
- actual/planned foot readback error max: `1.924e-6m`.

Remaining failures:

- root velocity error: `0.21598m/s`;
- joint step: `0.35389rad`;
- stance ground penetration: `1.872mm`;
- stance stationary ratio: `0.9667`;
- stance slip max: `5.120mm`;
- stance anchor residual: `5.225mm`.

## Rejected Interpretations

- A 160-cycle S4 trace is not a single-obstacle test; it reaches other course obstacles and produces unrelated collisions.
- The first sorted S4 anchor is a capsule; it produced a `142mm` ground-gap artifact and is not a stable representative.
- A direction-edge cuboid collides with the target itself in the 40-cycle crossing window and cannot be used to manufacture a pass.
- Moderate `0.4m/s` did not improve stance behavior; it produced about `8.4mm` maximum stance slip.
- The final sphere stance failure is not readback lag because actual/planned foot error remains micrometric.

## Result

Sharding infrastructure is ready, but the complete `29,640`-cell matrix is unrun. Real viewer small plumbing is complete and behavior-red on crossing-post-gait stance. Task 14 Steps 4-5 remain open; Stage B remains blocked.

## Scanner-Parity Follow-Up

- The formal fixture default was corrected from `0.05m` to the real scanner resolution `0.01m`; its focused tests pass (`14 passed`).
- The prior ranked small `7/7` is superseded. At `0.01m`, ranked small is `5/7`: translation and zero-command cells pass; only pure yaw is red (`+yaw` zero drift `1.1782e-5m`; `-yaw` zero drift `1.6495e-5m`, joint step `0.37055rad`).
- A second fixture mismatch is now isolated. `semantic_course.shape_params_for_profile()` creates the small sphere with `radius=0.06m`, so its grounded scanner peak is `0.12m`, while the formal sphere profile peaks at the nominal class height `0.16m`. The formal capsule is also flat-topped instead of using its native rounded cap.
- Geometry-parity tests and a fixture-only correction are required before further planner tuning or formal sharding. The representative viewer stance result remains independently red.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`, `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_backend_wiring.py`

## Follow-Up

First close formal/native shape geometry parity, then rerun ranked `0.01m` small. After that, trace the first post-crossing stance-anchor divergence against Contact and Terrain residuals on the same real scanner field. Keep H30, one RTI, seven losses, five alphas, and existing KKT/filter contracts fixed. Do not shorten the event window or weaken thresholds.
