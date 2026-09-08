"""AME CNN-attention actor critic adapted to a six-channel 16x16 map."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(current, width), nn.ELU()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class ActorCriticAME(nn.Module):
    """AME terrain encoder with shared CNN/MHA and separate actor/critic heads."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        map_channels: int = 6,
        map_size: int = 16,
        mha_dim: int = 64,
        num_heads: int = 16,
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        kwargs.pop("activation", None)
        if kwargs:
            raise TypeError(f"Unexpected ActorCriticAME arguments: {sorted(kwargs)}")
        super().__init__()
        self.map_channels = int(map_channels)
        self.map_size = int(map_size)
        self.map_dim = self.map_channels * self.map_size * self.map_size
        self.actor_state_dim = int(num_actor_obs) - self.map_dim
        self.critic_state_dim = int(num_critic_obs) - self.map_dim
        if self.actor_state_dim <= 0 or self.critic_state_dim <= 0:
            raise ValueError(
                "AME observations must contain a six-channel map followed by state; "
                f"got actor={num_actor_obs}, critic={num_critic_obs}, map={self.map_dim}"
            )

        self.map_cnn = nn.Sequential(
            nn.Conv2d(self.map_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, mha_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mha_dim),
        )
        self.actor_state_projection = nn.Linear(self.actor_state_dim, mha_dim)
        self.critic_state_projection = nn.Linear(self.critic_state_dim, mha_dim)
        self.mha = nn.MultiheadAttention(embed_dim=mha_dim, num_heads=num_heads, batch_first=True)
        self.actor = _mlp(mha_dim + self.actor_state_dim, num_actions, tuple(actor_hidden_dims))
        self.critic = _mlp(mha_dim + self.critic_state_dim, 1, tuple(critic_hidden_dims))
        self.std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        print(
            "[ActorCriticAME] map=(%d,%d,%d), actor_state=%d, critic_state=%d, actions=%d"
            % (
                self.map_channels,
                self.map_size,
                self.map_size,
                self.actor_state_dim,
                self.critic_state_dim,
                num_actions,
            )
        )

    def _encode(self, observations: torch.Tensor, *, critic: bool) -> torch.Tensor:
        expected = self.map_dim + (self.critic_state_dim if critic else self.actor_state_dim)
        if observations.ndim != 2 or observations.shape[1] != expected:
            raise ValueError(f"Expected AME observation [N,{expected}], got {tuple(observations.shape)}")
        map_tensor = observations[:, : self.map_dim].reshape(
            observations.shape[0], self.map_channels, self.map_size, self.map_size
        )
        state = observations[:, self.map_dim :]
        features = self.map_cnn(map_tensor).flatten(2).transpose(1, 2)
        projection = self.critic_state_projection if critic else self.actor_state_projection
        query = projection(state).unsqueeze(1)
        attended, _ = self.mha(query=query, key=features, value=features, need_weights=False)
        return torch.cat((attended.squeeze(1), state), dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(self._encode(observations, critic=False))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self._encode(observations, critic=False))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(self._encode(critic_observations, critic=True))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None) -> None:
        return None

    @torch.no_grad()
    def clip_std(self, min=None, max=None) -> None:
        self.std.copy_(self.std.clamp(min=min, max=max))


__all__ = ["ActorCriticAME"]
