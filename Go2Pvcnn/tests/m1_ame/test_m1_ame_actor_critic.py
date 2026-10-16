import torch
import torch.nn as nn

from m1_ame.m1_actor_critic_ame import M1ActorCriticAME


class _NonFiniteActor(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, features):
        return torch.full(
            (features.shape[0], self.num_actions),
            float("nan"),
            dtype=features.dtype,
            device=features.device,
        )


def test_ame_sampling_remains_valid_if_optimizer_temporarily_makes_std_negative():
    model = M1ActorCriticAME(
        num_actor_obs=6 * 4 * 4 + 2,
        num_critic_obs=6 * 4 * 4 + 3,
        num_actions=3,
        map_channels=6,
        map_size=4,
        mha_dim=4,
        num_heads=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
    )
    observations = torch.randn(2, 6 * 4 * 4 + 2)

    with torch.no_grad():
        model.std.fill_(-0.25)

    actions = model.act(observations)

    assert actions.shape == (2, 3)
    assert torch.isfinite(actions).all()
    assert torch.all(model.action_std > 0.0)


def test_ame_sampling_does_not_let_nonfinite_actor_mean_poison_distribution_scale():
    model = M1ActorCriticAME(
        num_actor_obs=6 * 4 * 4 + 2,
        num_critic_obs=6 * 4 * 4 + 3,
        num_actions=3,
        map_channels=6,
        map_size=4,
        mha_dim=4,
        num_heads=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
    )
    model.actor = _NonFiniteActor(num_actions=3)
    observations = torch.randn(2, 6 * 4 * 4 + 2)

    actions = model.act(observations)

    assert torch.isfinite(actions).all()
