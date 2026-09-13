# Parallelism AMP latest long-run diagnosis

## Purpose

Attribute behavior degradation in the latest completed AMP TensorBoard run to
the policy, value, discriminator, or planner-validity path.

## Input

- Run: `logs/rsl_rl/parallelism_tracking_cross_large_complex_amp/2026-08-28_18-54-54/4161d7b`
- Event file: `events.out.tfevents.1787914528.tj-a20978341498712064335212-worker-0.7.0`
- Checkpoint: `model_3499.pt`
- Config: `num_steps_per_env=40`, `num_learning_epochs=5`, `num_mini_batches=4`,
  `learning_rate=0.001`, `entropy_coef=0.01`, AMP weight target `0.1`.

## Evidence

- `Policy/mean_noise_std`: `0.3946` at step 0, `0.6509` at 500,
  `0.7579` at 600, `1.4465` at 1000, `7.5113` at 2000, peak `13.6619`
  near 2600, and `4.9914` at 3499.
- `AMP/amp_actor_reward_weight`: zero through step 500, `0.05` at 550,
  `0.1` from step 600 onward.
- `AMP/amp_value_loss` (AMP critic): maximum `167.69`; interval means remain
  roughly `2..4`.
- `AMP/value_loss` (base critic, returned under generic `value_loss`): maximum
  `1.127e6`; base returns become highly nonstationary after terminations and
  large negative episode returns.
- Expert and agent discriminator accuracy reach approximately `1.0` by step
  100 and stay saturated; D loss remains around `0.3..0.4`.
- `Train/mean_episode_length` ends at `3.1`; final
  `Episode_Termination/valid_bad_orientation` is `0.963`, with nonzero valid
  base-contact termination. This is physical/behavioral collapse, not NaN.
- Reward decomposition shows action-rate and joint/limit penalties dominate bad
  intervals (action-rate `-148.3` at step 1385 and `-56.1` at step 2300).
  All model, optimizer, D, and D optimizer tensors in `model_3499.pt` are finite.

## Code attribution

- `ParallelismAMPPPO.update()` performs the custom actor update but does not
  include standard PPO's adaptive-KL block or post-update `clip_std()` call.
- `ActorCriticCNN.std` is a raw trainable parameter used directly as the Normal
  scale; the AMP config keeps `entropy_coef=0.01`.
- The AMP override's `AMP/value_loss` metric is the base critic loss; only
  `AMP/amp_value_loss` is the AMP critic loss.

## Conclusion

The strongest supported cause is unbounded exploration-scale growth combined
with missing PPO KL/std controls in the AMP override. This increases action
variance, causes orientation/contact failures and short episodes, and then
makes the base critic loss explode because its returns are nonstationary. The
saturated discriminator and normalized AMP advantage are a secondary signal
quality problem after iteration 600, not evidence that the AMP critic itself
caused the collapse.

## Unverified

The run does not log AMP reward magnitude, AMP active ratio, or KL, so the exact
relative contribution of D guidance versus missing PPO controls cannot be
quantified from this event file alone. A controlled A/B run is required.

## Git ref

Code/design ref: `4161d7b`; this is a read-only diagnosis with no production
code change.
