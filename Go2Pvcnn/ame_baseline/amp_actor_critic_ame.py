"""AME actor-critic with a detached AMP value channel."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .actor_critic_ame import ActorCriticAME


class AmpActorCriticAME(ActorCriticAME):
    """Keep the AME policy/base critic and add an isolated AMP critic head.

    The AME map encoder is shared by the actor and base critic.  The AMP value
    head consumes a detached base-critic feature, so its loss cannot change the
    AME representation.  ``evaluate`` caches the feature for the immediately
    following ``evaluate_amp`` call, avoiding a third BatchNorm forward pass.
    """

    def __init__(self, *args, amp_value_hidden_dims=(256, 128), **kwargs):
        super().__init__(*args, **kwargs)
        critic_feature_dim = self.mha.embed_dim + self.critic_state_dim
        layers: list[nn.Module] = []
        current = critic_feature_dim + 2
        for hidden_dim in amp_value_hidden_dims:
            layers.extend((nn.Linear(current, int(hidden_dim)), nn.ELU()))
            current = int(hidden_dim)
        layers.append(nn.Linear(current, 1))
        self.amp_value_head = nn.Sequential(*layers)
        for layer in self.amp_value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        final = next(layer for layer in reversed(self.amp_value_head) if isinstance(layer, nn.Linear))
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self._critic_cache_observations: Tensor | None = None
        self._critic_cache_features: Tensor | None = None
        self.critic_feature_forward_count = 0

    def _encode_critic_for_value(self, observations: Tensor) -> Tensor:
        features = super()._encode(observations, critic=True)
        self._critic_cache_observations = observations
        self._critic_cache_features = features
        self.critic_feature_forward_count += 1
        return features

    def evaluate(self, critic_observations: Tensor, **kwargs) -> Tensor:
        """Evaluate the base environment value and cache its encoder feature."""

        return self.critic(self._encode_critic_for_value(critic_observations))

    def _cached_critic_features(self, observations: Tensor) -> Tensor:
        cached = self._critic_cache_features
        cached_observations = self._critic_cache_observations
        if (
            cached is None
            or cached_observations is None
            or cached_observations is not observations
            or cached.shape[0] != observations.shape[0]
        ):
            cached = self._encode_critic_for_value(observations)
        return cached

    def evaluate_amp(
        self,
        critic_observations: Tensor,
        amp_active: Tensor,
        history_ratio: Tensor,
    ) -> Tensor:
        """Evaluate ``V_amp`` from detached AME features and AMP context."""

        features = self._cached_critic_features(critic_observations).detach()
        active = torch.as_tensor(amp_active, device=features.device, dtype=features.dtype).reshape(-1, 1)
        ratio = torch.as_tensor(history_ratio, device=features.device, dtype=features.dtype).reshape(-1, 1)
        if active.shape[0] != features.shape[0] or ratio.shape[0] != features.shape[0]:
            raise ValueError(
                "AMP context batch must match critic observations: "
                f"features={features.shape[0]}, active={active.shape[0]}, ratio={ratio.shape[0]}"
            )
        return self.amp_value_head(torch.cat((features, active, ratio), dim=-1))

    def load_common_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Load a legacy AME checkpoint while leaving the AMP head initialized."""

        current = super().state_dict()
        filtered: dict[str, Tensor] = {}
        missing: list[str] = []
        mismatched: list[str] = []
        for key, value in current.items():
            if key.startswith("amp_value_head."):
                continue
            if key not in state_dict:
                missing.append(key)
                continue
            source = state_dict[key]
            if tuple(source.shape) != tuple(value.shape):
                mismatched.append(f"{key}: checkpoint {tuple(source.shape)} != model {tuple(value.shape)}")
                continue
            filtered[key] = source
        if missing or mismatched:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if mismatched:
                details.append("shape=" + ";".join(mismatched))
            raise ValueError("Incompatible AME checkpoint: " + " ".join(details))
        merged = dict(current)
        merged.update(filtered)
        super().load_state_dict(merged, strict=True)


__all__ = ["AmpActorCriticAME"]
