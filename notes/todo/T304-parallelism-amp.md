# T304 Parallelism AMP

## Current State

The isolated `parallelism_tracking_cross_large_complex_amp` experiment is
implemented on branch `parallelism-amp`. It adds a 39-D joint/root state frame,
23-transition batched history for a 24-frame discriminator window, `B0 -> B1`
replan alignment, standstill/reset gating, dual value channels, discriminator
replay/update, and legacy/full checkpoint resume.

## Open Children

- Long-run AMP quality and reward-weight tuning remain open; this task validates
  the training data path and stability smoke.

## Closed Children Archive

- Batch transition delta extraction and fixed-shape ring-buffer reconstruction.
- Real Isaac Lab smoke at 1024 environments for four PPO iterations.

## Related Logs

- [2026-08-27 AMP batched transition and smoke](../log/2026-08-27-parallelism-amp-batched-transition-and-1024-smoke.md)

## Git Refs

- Current work ref: `parallelism-amp`
- Last feature commit: `b8b6ab4`
- Last verified ref: `b8b6ab4`; focused `22 passed`

## Next Step

Use the AMP launcher with a longer resume run after selecting reward/value
coefficients. Distillation and pure PPO launchers remain unchanged.

## Node Details

Transition increments are computed for all environment rows with Torch/GPU
operations. Invalid rows are masked without CPU round-trips or per-environment
Python loops; only valid windows enter the discriminator replay queue.
