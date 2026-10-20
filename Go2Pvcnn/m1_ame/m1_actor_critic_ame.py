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
    _MAX_ACTION_STD = 0.40

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
            nan=0.10,
            posinf=self._MAX_ACTION_STD,
            neginf=self._MIN_ACTION_STD,
        ).clamp(min=self._MIN_ACTION_STD, max=self._MAX_ACTION_STD)
        scale = torch.ones_like(mean) * std
        self.distribution = Normal(mean, scale)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self.actor(self._encode(observations, critic=False))
        return torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def clip_std(self, min=None, max=None) -> None:
        """Keep M1 exploration bounded after every PPO update."""

        self.std.copy_(
            torch.nan_to_num(
                self.std,
                nan=0.10,
                posinf=self._MAX_ACTION_STD,
                neginf=self._MIN_ACTION_STD,
            ).clamp(min=self._MIN_ACTION_STD, max=self._MAX_ACTION_STD)
        )


__all__ = ["M1ActorCriticAME"]
