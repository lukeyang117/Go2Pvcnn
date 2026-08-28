# Parallelism AMP actor warm-up and real smoke

## Purpose

Verify the post-NaN AMP normalization fix and the 500-iteration Actor guidance
warm-up while preserving the existing pure PPO and distillation routes.

## Stage

AMP PPO algorithm, masked rollout storage, runner iteration schedule, and the
isolated Parallelism AMP training configuration.

## Procedure

- Focused tests:
  `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_*.py Go2Pvcnn/tests/tracking/test_parallelism_amp_*.py`
- Compile and whitespace checks:
  `python -m compileall` on the five changed Python modules and `git diff --check`.
- Real smoke:
  `Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py` with the
  pure PPO checkpoint
  `/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/cross_large_complex_ppo/2026-08-26_17-47-24/11d453a/model_19998.pt`,
  `--num-envs 1024`, and `--max-iterations 4`.
- TensorBoard event inspection of
  `logs/rsl_rl/parallelism_tracking_cross_large_complex_amp/2026-08-28_18-44-41/cecbc73`.

## Results

- Focused AMP suite: `24 passed in 11.39s`.
- Compile exit `0`; `git diff --check` exit `0`.
- Real Isaac Lab process exit `0`; iterations `0..3` completed.
- Legacy policy warm-start initialized `amp_value_head` successfully.
- TensorBoard contains AMP value, discriminator, and actor-weight scalars at
  steps `0..3`; `AMP/amp_actor_reward_weight` is `0.0` during this pre-ramp run.
- No `Traceback`, `NaN`, `Inf`, `OutOfMemory`, or OOM marker was present.

## Conclusion

The Actor schedule is wired to the resumed global iteration, while AMP Critic
and discriminator training remain active from iteration 0. The single-active
sample AMP GAE path is finite because normalization uses population standard
deviation (`unbiased=False`).

## Follow-up

Run a longer resume after iteration 500 to observe the `500..600` ramp and
long-run AMP reward/value stability.

## Git refs

- Baseline ref: `cecbc73`
- Candidate/last verified ref: `7242c08`
- Key files: `parallelism_amp_ppo.py`, `parallelism_amp_storage.py`,
  `on_policy_runner.py`, `train_cfg.py`, AMP env config, and AMP tests.
