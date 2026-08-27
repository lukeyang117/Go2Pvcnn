# Parallelism AMP batched transition and 1024-env smoke

The AMP path computes agent/expert transition increments for the complete
environment batch with Torch tensors, keeps replan boundaries on `B0 -> B1`,
and feeds active 24-frame windows to the discriminator. Invalid/standstill rows
clear history and contribute zero AMP reward, advantage, value loss, and D
samples.

Verification: focused AMP tests `22 passed`; compilation and `git diff --check`
passed; real Isaac Lab `1024 env x 4 iterations` via the AMP launcher exited
with code `0`. TensorBoard run
`logs/rsl_rl/parallelism_tracking_cross_large_complex_amp/2026-08-27_21-18-02/b6ab22c`
contains discriminator loss/accuracy and value-loss scalar steps `0,1,2,3`,
with no OOM, NaN, Inf, or Traceback. The run used a legacy pure-PPO checkpoint;
AMP weights were initialized by `load_amp`.

Git refs: baseline `b6ab22c`; candidate/last verified `b8b6ab4`.
