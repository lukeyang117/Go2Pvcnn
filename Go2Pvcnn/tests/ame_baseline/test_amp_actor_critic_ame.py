from __future__ import annotations

import torch

from ame_baseline.actor_critic_ame import ActorCriticAME
from ame_baseline.amp_actor_critic_ame import AmpActorCriticAME


def _model(cls):
    return cls(
        1581,
        1584,
        12,
        map_channels=6,
        map_size=16,
        mha_dim=64,
        num_heads=16,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
    )


def test_amp_head_and_context_shape():
    model = _model(AmpActorCriticAME)
    critic = torch.randn(8, 1584)
    value = model.evaluate_amp(critic, torch.ones(8), torch.ones(8))
    assert value.shape == (8, 1)
    assert torch.equal(value, torch.zeros_like(value))


def test_amp_value_loss_does_not_update_ame_encoder():
    model = _model(AmpActorCriticAME)
    value = model.evaluate_amp(torch.randn(8, 1584), torch.ones(8), torch.ones(8))
    value.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.map_cnn.parameters())
    assert any(parameter.grad is not None for parameter in model.amp_value_head.parameters())


def test_legacy_ame_state_is_loaded_without_amp_keys():
    torch.manual_seed(17)
    source = _model(ActorCriticAME)
    target = _model(AmpActorCriticAME)
    target.load_common_state_dict(source.state_dict())
    target_state = target.state_dict()
    for key, value in source.state_dict().items():
        assert torch.equal(target_state[key], value), key
    assert any(key.startswith("amp_value_head.") for key in target_state)


def test_evaluate_then_evaluate_amp_reuses_one_critic_feature():
    model = _model(AmpActorCriticAME)
    critic = torch.randn(8, 1584)
    model.evaluate(critic)
    first = model.critic_feature_forward_count
    model.evaluate_amp(critic, torch.ones(8), torch.ones(8))
    assert model.critic_feature_forward_count == first
