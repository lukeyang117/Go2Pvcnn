from types import SimpleNamespace

import pytest
import torch

from rsl_rl.algorithms.parallelism_amp_ppo import ParallelismAMPPPO
from rsl_rl.modules.amp_actor_critic_cnn import AmpActorCriticCNN
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.runners.on_policy_runner import IncompleteAMPCheckpointError, OnPolicyRunner


def _runner():
    model = AmpActorCriticCNN(8, 8, 2, use_cost_map=False, actor_hidden_dims=[16], critic_hidden_dims=[16])
    alg = ParallelismAMPPPO(model, num_learning_epochs=1, num_mini_batches=1, device="cpu")
    runner = object.__new__(OnPolicyRunner)
    runner.alg = alg
    runner.device = "cpu"
    runner.empirical_normalization = False
    runner.current_learning_iteration = 0
    return runner


def test_legacy_checkpoint_preserves_base_outputs_and_zeroes_amp_head(tmp_path):
    source = _runner()
    sample = torch.randn(2, 8)
    baseline = source.alg.actor_critic.evaluate(sample).detach()
    checkpoint = tmp_path / "pure.pt"
    torch.save({"model_state_dict": {k: v for k, v in source.alg.actor_critic.state_dict().items() if not k.startswith("amp_value_head.")}, "iter": 17}, checkpoint)
    target = _runner()
    mode = target.load_amp(checkpoint, keep_std=True)
    assert mode == "legacy_policy_warm_start"
    assert torch.allclose(target.alg.actor_critic.evaluate(sample), baseline)
    assert torch.equal(target.alg.actor_critic.evaluate_amp(sample, torch.ones(2), torch.ones(2)), torch.zeros(2, 1))
    assert target.current_learning_iteration == 0


def test_half_amp_checkpoint_is_rejected(tmp_path):
    source = _runner()
    checkpoint = tmp_path / "partial.pt"
    torch.save({"model_state_dict": source.alg.actor_critic.state_dict(), "iter": 1}, checkpoint)
    with pytest.raises(IncompleteAMPCheckpointError):
        _runner().load_amp(checkpoint)


def test_full_amp_checkpoint_restores_iteration_and_discriminator(tmp_path):
    source = _runner()
    source.alg.amp_discriminator = AMPDiscriminator()
    checkpoint = tmp_path / "amp.pt"
    torch.save({
        "model_state_dict": source.alg.actor_critic.state_dict(),
        "amp_discriminator_state_dict": source.alg.amp_discriminator.state_dict(),
        "amp_optimizer_state_dict": source.alg.amp_discriminator.optimizer.state_dict(),
        "iter": 23,
    }, checkpoint)
    target = _runner()
    assert target.load_amp(checkpoint) == "full_amp_resume"
    assert target.current_learning_iteration == 23
    assert target.alg.amp_discriminator is not None
