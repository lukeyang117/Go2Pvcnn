"""Actor-critic CNN with a detached, separately trained AMP value head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .actor_critic_cnn import ActorCriticCNN, get_activation


class AmpActorCriticCNN(ActorCriticCNN):
    """Keep the legacy actor/base critic contract and add ``V_amp``."""

    def __init__(self, *args, amp_value_hidden_dims=(256, 128), activation="elu", **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
        critic_feature_dim = int(self.critic[0].in_features)
        layers: list[nn.Module] = []
        current = critic_feature_dim + 2
        for hidden in amp_value_hidden_dims:
            layers.extend((nn.Linear(current, int(hidden)), get_activation(activation).__class__()))
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.amp_value_head = nn.Sequential(*layers)
        for layer in self.amp_value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        final = [layer for layer in self.amp_value_head if isinstance(layer, nn.Linear)][-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def evaluate_amp(self, critic_observations: Tensor, amp_active: Tensor, history_ratio: Tensor) -> Tensor:
        features = self._extract_critic_features(critic_observations).detach()
        active = torch.as_tensor(amp_active, device=features.device, dtype=features.dtype).reshape(-1, 1)
        ratio = torch.as_tensor(history_ratio, device=features.device, dtype=features.dtype).reshape(-1, 1)
        if active.shape[0] != features.shape[0] or ratio.shape[0] != features.shape[0]:
            raise ValueError("AMP context batch must match critic observations")
        return self.amp_value_head(torch.cat((features, active, ratio), dim=-1))

    def load_common_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load only legacy actor/base-critic keys with strict shape checks."""

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
            if tuple(state_dict[key].shape) != tuple(value.shape):
                mismatched.append(f"{key}: checkpoint {tuple(state_dict[key].shape)} != model {tuple(value.shape)}")
                continue
            filtered[key] = state_dict[key]
        if missing or mismatched:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if mismatched:
                details.append("shape=" + ";".join(mismatched))
            raise ValueError("Incompatible legacy ActorCriticCNN checkpoint: " + " ".join(details))
        merged = dict(current)
        merged.update(filtered)
        super().load_state_dict(merged, strict=True)
