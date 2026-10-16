"""M1 AME policy with a numerical guard for PPO action-noise sampling."""

from __future__ import annotations

import torch
from torch.distributions import Normal

from ame_baseline.actor_critic_ame import ActorCriticAME


class M1ActorCriticAME(ActorCriticAME):
    """M1 AME network with the Go2 architecture and safe action sampling."""

    _MIN_ACTION_STD = 1.0e-15

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(self._encode(observations, critic=False))
        # Keep the normal distribution valid even if a late PPO update
        # temporarily produces a non-finite actor output.  This is a
        # distribution-boundary guard only; finite actor outputs are unchanged.
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        # PPO clips the trainable parameter after an update.  During a
        # multi-minibatch update it can temporarily cross zero between two
        # minibatches, so clamp only the distribution view here.
        std = torch.nan_to_num(
            self.std,
            nan=1.0,
            posinf=1.0,
            neginf=self._MIN_ACTION_STD,
        ).clamp_min(self._MIN_ACTION_STD)
        scale = torch.ones_like(mean) * std
        self.distribution = Normal(mean, scale)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.actor(self._encode(observations, critic=False))
        return torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = ["M1ActorCriticAME"]
