# T303 Parallelism Flat Foot Planner

## Status

verify

## Summary

`extension/parallelism` now contains a self-contained flat/highmap Go2 foot planner for the `Parallelism` branch.

## Implemented

- Batched public contracts for state, terrain, diagnostics, trajectory, and RL reference.
- Isaac-style height/semantic/valid terrain query path.
- Go2 constants, batched FK, analytic IK, root smoothstep rollout, and per-foot golden-angle candidate sampling.
- Single-pass torch hard filters and velocity-tracking score with one `argmin`.
- 24-frame fixed trot trajectory assembly and shape-only RL adapter.
- `go2_foostep_planner.py --planner-backend parallelism` route and viewer adapter.

## Verification

- `14 passed`: `Go2Pvcnn/tests/parallelism` plus `Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py`.
- Import smoke prints `24` for `ParallelismCfg().horizon`.

## Follow-Up

Run a real Isaac viewer smoke with:

`Go2Pvcnn/extension/viz/go2_foostep_planner.py --planner-backend parallelism --device cuda:0 --num_envs 1 --terrain task --n-frames 30 --plan-dt 0.02 --terrain-row 0 --terrain-col 0 --warmup-steps 0 --key-hold-timeout 3.0`
