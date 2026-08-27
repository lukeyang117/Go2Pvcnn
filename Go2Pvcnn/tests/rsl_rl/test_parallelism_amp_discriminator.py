import torch

from rsl_rl.modules.amp_discriminator import AMPDiscriminator


def test_discriminator_step_changes_parameters_and_zeroes_inactive_reward():
    disc = AMPDiscriminator(input_dim=936, hidden_dims=(32, 16))
    expert = torch.randn(8, 936)
    agent = torch.randn(8, 936)
    active = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
    before = [p.detach().clone() for p in disc.parameters()]
    metrics = disc.update(expert, agent, active)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, disc.parameters()))
    rewards = disc.reward(agent.reshape(8, 24, 39), active)
    assert torch.equal(rewards[4:], torch.zeros(4))


def test_normalizer_ignores_inactive_rows():
    disc = AMPDiscriminator(input_dim=4, hidden_dims=(8,))
    windows = torch.tensor([[1.0, 1.0, 1.0, 1.0], [100.0, 100.0, 100.0, 100.0]])
    disc.normalizer.update(windows, torch.tensor([True, False]))
    assert torch.allclose(disc.normalizer.mean, torch.ones(4), atol=1e-5)
