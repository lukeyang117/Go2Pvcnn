import torch

from rsl_rl.algorithms.parallelism_amp_ppo import ParallelismAMPPPO
from rsl_rl.modules.amp_actor_critic_cnn import AmpActorCriticCNN


def _make_algorithm_and_storage(active_rows=(True, False)):
    model = AmpActorCriticCNN(
        8,
        8,
        2,
        use_cost_map=False,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        amp_value_hidden_dims=[16],
    )
    alg = ParallelismAMPPPO(
        model,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
    )
    num_envs, horizon = 2, 2
    alg.init_storage(num_envs, horizon, [8], [8], [2])
    obs = torch.randn(num_envs, 8)
    active = torch.tensor(active_rows, dtype=torch.bool)
    for _ in range(horizon):
        alg.act(obs, obs, amp_context=(active, active.float()))
        alg.transition.rewards = torch.ones(num_envs)
        alg.transition.dones = torch.zeros(num_envs, dtype=torch.bool)
        alg.transition.amp_reward = active.float()
        alg.transition.amp_active = active
        alg.transition.history_ratio = active.float()
        alg.storage.add_transitions(alg.transition)
    alg.compute_returns(obs)
    return alg, alg.storage


def test_amp_update_uses_adaptive_kl_and_shared_learning_rate():
    alg, _ = _make_algorithm_and_storage()
    calls = []

    def record_clip_std(*, min=None, max=None):
        calls.append((min, max))

    alg.actor_critic.clip_std = record_clip_std
    metrics = alg.update()

    assert metrics["approx_kl"] >= 0.0
    assert alg.optimizer.param_groups[0]["lr"] == alg.learning_rate
    assert calls == [(alg.clip_min_std, None)]


def test_amp_value_loss_is_masked_and_reports_active_count():
    alg, storage = _make_algorithm_and_storage(active_rows=(True, False))
    storage.amp_values.fill_(100.0)
    storage.amp_returns.zero_()
    metrics = alg.update()

    assert torch.isfinite(torch.tensor(metrics["amp_value_loss"]))
    assert metrics["amp_active_count"] == 2


def test_amp_update_reports_clipped_finite_actor_critic_gradient_norm():
    alg, _ = _make_algorithm_and_storage()
    metrics = alg.update()

    assert torch.isfinite(torch.tensor(metrics["actor_critic_grad_norm"]))
    assert torch.isfinite(torch.tensor(metrics["actor_critic_grad_norm_clipped"]))
    assert metrics["actor_critic_grad_norm_clipped"] <= alg.max_grad_norm + 1.0e-5
