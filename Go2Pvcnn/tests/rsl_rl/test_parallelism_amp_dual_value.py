import torch

from rsl_rl.modules import ActorCriticCNN
from rsl_rl.modules.amp_actor_critic_cnn import AmpActorCriticCNN


def _kwargs():
    return {"use_cost_map": False, "actor_hidden_dims": [16], "critic_hidden_dims": [16]}


def test_amp_value_gradient_does_not_reach_base_network():
    model = AmpActorCriticCNN(8, 8, 4, **_kwargs())
    obs = torch.randn(6, 8)
    loss = model.evaluate_amp(obs, torch.ones(6), torch.ones(6)).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.amp_value_head.parameters())
    assert all(p.grad is None for p in model.critic.parameters())


def test_base_state_dict_keys_and_output_match_actor_critic_cnn():
    base = ActorCriticCNN(8, 8, 4, **_kwargs())
    amp = AmpActorCriticCNN(8, 8, 4, **_kwargs())
    amp.load_common_state_dict(base.state_dict())
    sample = torch.ones(2, 8)
    assert torch.allclose(base.act_inference(sample), amp.act_inference(sample))
    assert torch.allclose(base.evaluate(sample), amp.evaluate(sample))
