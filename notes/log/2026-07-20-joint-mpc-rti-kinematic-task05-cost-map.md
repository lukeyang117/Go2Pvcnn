# Joint MPC RTI Kinematic Task 05 Cost Map

## Purpose

Add spatially differentiable semantic observations while preserving exact signed fields and raw detector inputs.

## Stage

- Todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Plan task: 5 of 16

## TDD Evidence

RED: four focused tests failed on the missing `terrain.cost_map` module and missing terrain config.

Focused GREEN: `4 passed, 23 deselected in 3.53s`.

Combined Tasks 1-5 CPU command used `CUDA_VISIBLE_DEVICES=''` and ran terrain, contract, gait, IK, nominal, FK, and Jacobian tests.

Result: `37 passed, 13 skipped in 5.61s`.

## Contract

- Small/large masks and semantic-weighted height use one fixed four-group Gaussian `conv2d`.
- Occupancy is `1-exp(-gain*mass)` and propagated class height is weighted-height mass divided by occupancy mass.
- Fixed grouped Scharr convolution produces explicit local XY occupancy gradients.
- World query bilinearly samples soft occupancy, propagated height, and gradients and remains differentiable with respect to query XY.
- Small swing uses propagated physical height; small foot stance and all large queries can use continuous `h_wall`.
- Exact signed small/large distance, raw semantic ID, raw elevation, and physical geometry remain available for acceptance detectors.
- Direct terrain construction and mutable cache both publish the new soft fields.
- Optimizer-tunable convolution parameters are `small_sigma_m`, `large_sigma_m`, `small_gain`, `large_gain`, and `h_wall`; kernel shape and Scharr structure are fixed.

## Known Environment Limit

The 13 CUDA exact-EDT cases were skipped in this CPU pass. The initial GPU-visible baseline failed allocations because the current device was out of memory. CUDA field/cache validation remains required under the monitored runner when sufficient memory is available.

## Git Refs

- Baseline Ref: `97f498d`
- Candidate Ref: `work/joint-mpc-kinematic`
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/terrain/cost_map.py`, `Go2Pvcnn/extension/joint_mpc_rti/terrain/query.py`, `Go2Pvcnn/extension/joint_mpc_rti/terrain/field_cache.py`

## Follow-Up

Replace the objective with exactly seven state-trajectory losses; terrain loss must not branch on raw semantic IDs.
