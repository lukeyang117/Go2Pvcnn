import torch

from rsl_rl.algorithms.parallelism_amp_ppo import ParallelismAMPPPO
from rsl_rl.modules.amp_actor_critic_cnn import AmpActorCriticCNN
from rsl_rl.modules.amp_discriminator import AMPDiscriminator


def test_amp_actor_weight_warms_up_after_global_iteration_500():
    model = AmpActorCriticCNN(8, 8, 2, use_cost_map=False, actor_hidden_dims=[16], critic_hidden_dims=[16])
    alg = ParallelismAMPPPO(
        model,
        amp_reward_weight=0.1,
        amp_warmup_iterations=500,
        amp_weight_ramp_iterations=100,
        device="cpu",
    )

    alg.set_iteration(0, 700)
    assert alg.actor_amp_reward_weight == 0.0
    alg.set_iteration(499, 700)
    assert alg.actor_amp_reward_weight == 0.0
    alg.set_iteration(500, 700)
    assert alg.actor_amp_reward_weight == 0.0
    alg.set_iteration(550, 700)
    assert alg.actor_amp_reward_weight == 0.05
    alg.set_iteration(600, 700)
    assert alg.actor_amp_reward_weight == 0.1
    alg.set_iteration(900, 1000)
    assert alg.actor_amp_reward_weight == 0.1


def test_one_update_is_finite_and_clears_rollout_only():
    model = AmpActorCriticCNN(8, 8, 2, use_cost_map=False, actor_hidden_dims=[16], critic_hidden_dims=[16])
    alg = ParallelismAMPPPO(model, num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-3, device="cpu")
    alg.amp_discriminator = AMPDiscriminator(hidden_dims=(16, 8), learning_rate=1e-3)
    alg.init_storage(4, 4, [8], [8], [2])
    obs = torch.randn(4, 8)
    for _ in range(4):
        alg.act(obs, obs)
        alg.transition.rewards = torch.ones(4)
        alg.transition.dones = torch.zeros(4, dtype=torch.bool)
        alg.transition.amp_reward = torch.ones(4)
        alg.transition.amp_active = torch.ones(4, dtype=torch.bool)
        alg.process_env_step(
            torch.ones(4),
            torch.zeros(4, dtype=torch.bool),
            {
                "amp": {
                    "amp_active": torch.ones(4),
                    "history_ratio": torch.ones(4),
                    "expert_window": torch.randn(4, 24, 39),
                    "agent_window": torch.randn(4, 24, 39),
                }
            },
        )
    alg.compute_returns(obs)
    metrics = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert metrics["discriminator_loss"] > 0.0
    assert alg.storage.step == 0
