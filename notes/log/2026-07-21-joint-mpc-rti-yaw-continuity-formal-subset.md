# Joint MPC RTI Yaw Continuity And Formal Subset

## Purpose

Close the Task 14 ranked pure-yaw regression, validate the corrected late-phase small fixture, and recheck flat behavior without changing the fixed H30/one-RTI/seven-loss architecture.

## Stage

- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Stage: Task 14 ranked small, ranked flat regression, and formal late-phase representative subset
- Device: `cuda:1` for behavior; CPU for pytest

## Root Causes

1. `shift_rebase_trajectory` wrapped every future yaw node independently. When the H30 tail crossed `+/-pi` before published `x1`, Command and Smooth saw a false `2*pi` edge and reversed yaw. FK/Contact coupling then produced the positive-yaw RP-rate and stance-slip failures.
2. The small acceptance field kept the obstacle-entry origin for all 160 steps. Late-phase `vx=1.0` traces reached the stale 151x151 map edge, coupling `map_valid_ratio`, ground-gap, and airborne-touchdown false failures.
3. After those defects were removed, one real continuing-stance cell exceeded the `0.5mm` segment-anchor gate. Raising the existing strong Contact subweight `contact_anchor_xy` from `200` to `400`, while retaining weak future onset `1`, closed it without opposing Terrain at touchdown.

## Changes

- Warm yaw rebase now applies one coordinate delta to the whole horizon and uses wrapped delta only for physical XY rotation.
- Small fixture rebuilds the local field around current measured root every step while preserving one fixed world obstacle center.
- Production `contact_anchor_xy=400`; `contact_future_onset_xy=1.0` remains unchanged.
- Approved design and implementation plan document both amendments.

## Verification

- Yaw continuity RED: one test failed with a `6.283185rad` false edge; GREEN: two crossing/branch tests passed.
- Focused nominal/loss/QP: `49 passed`.
- Small fixture suite: `13 passed`.
- Ranked small, 7 axis-isolated commands x 160 steps: `7/7`.
- Ranked flat, 7 axis-isolated commands x 144 steps: `7/7`.
- Late-phase subset: `vx=1.0`, phases `20/21/22`, sphere/cuboid, offsets `-0.04/+0.04`: `12/12`.
- Full `Go2Pvcnn/tests/joint_mpc_rti`: `205 passed in 43.38s`.

Pure-yaw ranked small after the fix:

| Command | Yaw-rate error | RP-rate max | Stance slip max | Segment residual |
| --- | ---: | ---: | ---: | ---: |
| `(0,0,+1)` | `0.037646rad/s` | `0.038354rad/s` | `0.0117mm` | `0.0117mm` |
| `(0,0,-1)` | `0.035484rad/s` | `0.041480rad/s` | `0.0106mm` | `0.0106mm` |

## Result

Ranked flat/small and the targeted formal late-phase subset are green under production defaults. The complete `29,640`-cell formal small matrix and real viewer small crossing remain unverified and open.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py`, `Go2Pvcnn/extension/joint_mpc_rti/config.py`, `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py`

## Follow-Up

Run the deterministic contiguous formal shards and exact-key merge gate, then run the real viewer small crossing. Do not start Stage B until Task 14 formal/viewer evidence is complete.
