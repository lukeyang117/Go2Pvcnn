"""M1 AME policy with a numerical guard for PPO action-noise sampling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from ame_baseline.actor_critic_ame import ActorCriticAME


class M1ActorCriticAME(ActorCriticAME):
    """M1 AME network with the Go2 architecture and safe action sampling."""

    _MIN_ACTION_STD = 0.03
    _LEG_MAX_ACTION_STD = 0.30
    _WHEEL_MAX_ACTION_STD = 0.10
    _ACTION_LIMIT = 1.0
    _ACTION_EPS = 1.0e-6
    _WHEEL_ACTION_INDICES = (3, 7, 11, 15)

    def __init__(self, *args, **kwargs):
        init_noise_std = kwargs.pop("init_noise_std", 1.0)
        if isinstance(init_noise_std, Sequence) and not isinstance(init_noise_std, (str, bytes)):
            values = torch.as_tensor(tuple(init_noise_std), dtype=torch.float32)
            num_actions = kwargs.get("num_actions", args[2] if len(args) > 2 else None)
            if num_actions is None or values.shape != (int(num_actions),):
                raise ValueError(
                    "M1 per-action init_noise_std must have one value per action: "
                    f"got {tuple(values.shape)} for {num_actions} actions"
                )
            if not torch.isfinite(values).all() or bool((values <= 0).any().item()):
                raise ValueError("M1 init_noise_std must contain finite positive values")
            super().__init__(*args, init_noise_std=1.0, **kwargs)
            self.std = nn.Parameter(values)
        else:
            super().__init__(*args, init_noise_std=init_noise_std, **kwargs)
        self._last_squashed_action: torch.Tensor | None = None

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(self._encode(observations, critic=False))
        # Keep the normal distribution valid even if a late PPO update
        # temporarily produces a non-finite actor output.  This is a
        # distribution-boundary guard only; finite actor outputs are unchanged.
        mean = torch.nan_to_num(mean, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        # Bound the base Gaussian per action type before the tanh transform.
        std = self._bounded_std()
        scale = torch.ones_like(mean) * std
        self.distribution = Normal(mean, scale)

    def _bounded_std(self) -> torch.Tensor:
        std = torch.nan_to_num(
            self.std,
            nan=0.10,
            posinf=self._LEG_MAX_ACTION_STD,
            neginf=self._MIN_ACTION_STD,
        ).clamp(min=self._MIN_ACTION_STD)
        std_max = torch.full_like(std, self._LEG_MAX_ACTION_STD)
        for index in self._WHEEL_ACTION_INDICES:
            if index < std.numel():
                std_max[index] = self._WHEEL_MAX_ACTION_STD
        return torch.minimum(std, std_max)

    @property
    def effective_action_std(self) -> torch.Tensor:
        return self._bounded_std()

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        self._last_squashed_action = torch.tanh(self.distribution.rsample())
        return self._last_squashed_action

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        bounded = actions.clamp(-self._ACTION_LIMIT + self._ACTION_EPS, self._ACTION_LIMIT - self._ACTION_EPS)
        latent = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        log_jacobian = torch.log((1.0 - bounded.square()).clamp_min(self._ACTION_EPS))
        return (self.distribution.log_prob(latent) - log_jacobian).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.actor(self._encode(observations, critic=False))
        return torch.tanh(torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0))

    @property
    def entropy(self) -> torch.Tensor:
        if self._last_squashed_action is None:
            return self.distribution.entropy().sum(dim=-1)
        return -self.get_actions_log_prob(self._last_squashed_action)

    @torch.no_grad()
    def clip_std(self, min=None, max=None) -> None:
        """Keep M1 exploration bounded after every PPO update."""

        self.std.copy_(self._bounded_std())


__all__ = ["M1ActorCriticAME"]
