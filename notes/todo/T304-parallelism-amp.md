# T304 Parallelism AMP

## Current State

The isolated `parallelism_tracking_cross_large_complex_amp` experiment is
implemented on branch `parallelism-amp`. It adds a 39-D joint/root state frame,
23-transition batched history for a 24-frame discriminator window, `B0 -> B1`
replan alignment, standstill/reset gating, dual value channels, discriminator
replay/update, and legacy/full checkpoint resume. AMP Critic and discriminator
training start at iteration 0; Actor AMP guidance stays at zero through
iteration 500, ramps linearly to 0.1 by iteration 600, and remains 0.1 after
that.

## Open Children

- Long-run AMP quality and reward-weight tuning remain open; this task validates
  the training data path and stability smoke.
- Long-run behavior collapse is reproduced in `2026-08-28_18-54-54/4161d7b`:
  action std grows without an upper bound, `valid_bad_orientation` reaches
  `0.963`, and the discriminator remains saturated near 100% expert/agent
  accuracy. Immediate hypotheses are missing std/KL control in the AMP PPO
  override and a non-informative saturated D signal; `AMP/amp_value_loss` is
  not the dominant exploding metric.

## Closed Children Archive

- Batch transition delta extraction and fixed-shape ring-buffer reconstruction.
- Real Isaac Lab smoke at 1024 environments for four PPO iterations.

## Related Logs

- [2026-08-27 AMP batched transition and smoke](../log/2026-08-27-parallelism-amp-batched-transition-and-1024-smoke.md)
- [2026-08-28 AMP warm-up and real smoke](../log/2026-08-28-parallelism-amp-warmup-real-smoke.md)
- [2026-08-31 AMP long-run diagnosis](../log/2026-08-31-parallelism-amp-long-run-diagnosis.md)

## Git Refs

- Current work ref: `parallelism-amp`
- Last feature commit: `7242c08`
- Last verified ref: `7242c08`; focused `24 passed`; real `1024 env x 4 iterations` smoke exit `0`

## Next Step

Before another long run, instrument AMP reward/active-ratio and KL, then test
the AMP PPO update against standard PPO KL/std controls. Keep reward/value
coefficient changes separate from optimizer-control changes. Distillation and
pure PPO launchers remain unchanged.

## Node Details

Transition increments are computed for all environment rows with Torch/GPU
operations. Invalid rows are masked without CPU round-trips or per-environment
Python loops; only valid windows enter the discriminator replay queue.
