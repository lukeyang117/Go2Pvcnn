# Joint MPC RTI Kinematic Nominal Anchor Fix

## Purpose

Trace and correct the flat ranked stance-slip failure without changing loss categories, KKT rows, scan, or line search.

## Root cause

At phase zero, active stance legs have `lift_raw < 0` but not `stance_raw < 0`. The nominal builder therefore replaced their measured stance anchors with nominal event footprints after node zero. The first QP moved a stance foot by about `0.13 m`. Warm nominal then discarded newly built foot references and used shifted-state FK, recreating the same phase-boundary error.

## Changes

- Current stance references use measured feet throughout the pre-lift interval (`lift_raw < 0`) while preserving the vectorized one-call nominal construction.
- Warm nominal exposes `references.foot` as `foot_reference_w`; it no longer discards the current measured stance reference in favor of shifted-state FK.
- Metrics compare root velocity to body-frame command using the same SE(2) transform as `command_residual`.
- Stance anchor metric uses XY only; ground Z gap/penetration remains a separate metric.

## Verification

- RED test reproduced current phase stance anchor mismatch.
- Nominal regression after fix: `9 passed`.
- Monitored CUDA ranked rerun completed in about `13.0s`; forward/backward stance slip reduced from about `0.134m` to below `0.001m`.
- Root direction/linear velocity false failures disappeared after body-frame metric correction.
- Remaining real failures: zero drift, swing/root event behavior, some stance residuals, backward joint/ground/line-search behavior. Flat gate remains open.

## Git refs

- Baseline ref: `4ed0ce9`
- Candidate ref: uncommitted before checkpoint commit
- Key files: `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py`, `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`
