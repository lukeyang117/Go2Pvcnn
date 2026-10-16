import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


class _NonFiniteValueActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(1.0))
        self._action_mean = torch.zeros(2, 1)
        self._action_std = torch.ones(2, 1)

    def act(self, observations, **kwargs):
        self._action_mean = torch.zeros(observations.shape[0], 1)
        self._action_std = torch.ones(observations.shape[0], 1)
        return torch.zeros(observations.shape[0], 1)

    def evaluate(self, observations, **kwargs):
        return self.parameter.expand(observations.shape[0], 1) * float("nan")

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0])

    @property
    def action_mean(self):
        return self._action_mean

    @property
    def action_std(self):
        return self._action_std

    @property
    def entropy(self):
        return torch.zeros(self._action_mean.shape[0])

    def reset(self, dones=None):
        return None

    def clip_std(self, min=None, max=None):
        return None


class _OneBatchStorage:
    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        zeros = torch.zeros(2, 1)
        yield (
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            torch.ones_like(zeros),
            (None, None),
            None,
            None,
            None,
        )

    def clear(self):
        return None


def test_ppo_skips_nonfinite_minibatch_without_corrupting_actor_critic():
    actor_critic = _NonFiniteValueActorCritic()
    algorithm = PPO(
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )
    algorithm.storage = _OneBatchStorage()
    before = actor_critic.parameter.detach().clone()

    value_loss, surrogate_loss = algorithm.update()

    assert torch.isfinite(actor_critic.parameter).all()
    assert torch.allclose(actor_critic.parameter, before)
    assert value_loss == 0.0
    assert surrogate_loss == 0.0
