from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ame_baseline.ame_amp_runner import AME_AMP_ARCHITECTURE_SIGNATURE, AmeAmpOnPolicyRunner
from ame_baseline.ame_amp_train_cfg import get_ame_amp_train_cfg
from ame_baseline.actor_critic_ame import ActorCriticAME


class _FakeVecEnv:
    num_envs = 2
    num_actions = 12
    max_episode_length = 100
    device = "cpu"
    cfg = {}

    def get_observations(self):
        return torch.zeros(self.num_envs, 1581), {"observations": {"critic": torch.zeros(self.num_envs, 1584)}}

    def step(self, actions):
        raise NotImplementedError


def _runner(tmp_path: Path) -> AmeAmpOnPolicyRunner:
    return AmeAmpOnPolicyRunner(_FakeVecEnv(), copy.deepcopy(get_ame_amp_train_cfg()), log_dir=str(tmp_path), device="cpu")


def _legacy_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    model = ActorCriticAME(1581, 1584, 12, map_channels=6, map_size=16, mha_dim=64, num_heads=16)
    state = model.state_dict()
    checkpoint = tmp_path / "ame_model_19999.pt"
    torch.save(
        {
            "model_state_dict": state,
            "iter": 19999,
            "ame_architecture_signature": AME_AMP_ARCHITECTURE_SIGNATURE,
            "ame_num_actor_obs": 1581,
            "ame_num_critic_obs": 1584,
            "ame_num_actions": 12,
        },
        checkpoint,
    )
    return checkpoint, state


def test_legacy_ame_checkpoint_warm_starts_without_old_optimizer(tmp_path):
    source_path, source_state = _legacy_checkpoint(tmp_path)
    runner = _runner(tmp_path)
    mode = runner.load_amp_checkpoint(source_path)
    assert mode == "legacy_policy_warm_start"
    assert runner.current_learning_iteration == 0
    assert torch.equal(runner.alg.actor_critic.std.detach(), source_state["std"])
    assert runner.alg.optimizer.state_dict()["state"] == {}
    assert runner.source_metadata["ame_source_iteration"] == 19999


def test_full_amp_resume_restores_both_optimizers(tmp_path):
    source_path, _ = _legacy_checkpoint(tmp_path)
    runner = _runner(tmp_path)
    runner.load_amp_checkpoint(source_path)
    loss = runner.alg.actor_critic.std.square().sum()
    runner.alg.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    runner.alg.optimizer.step()
    d_loss = sum(value.square().sum() for value in runner.alg.amp_discriminator.parameters())
    runner.alg.amp_discriminator.optimizer.zero_grad(set_to_none=True)
    d_loss.backward()
    runner.alg.amp_discriminator.optimizer.step()
    runner.current_learning_iteration = 7
    full_path = tmp_path / "model_7.pt"
    runner.save(full_path)

    resumed = _runner(tmp_path)
    assert resumed.load_amp_checkpoint(full_path) == "full_amp_resume"
    assert resumed.current_learning_iteration == 7
    assert resumed.alg.optimizer.state_dict()["state"]
    assert resumed.alg.amp_discriminator.optimizer.state_dict()["state"]


def test_partial_amp_checkpoint_is_rejected(tmp_path):
    source_path, source_state = _legacy_checkpoint(tmp_path)
    partial = dict(torch.load(source_path, map_location="cpu"))
    partial["model_state_dict"] = dict(source_state)
    partial["model_state_dict"]["amp_value_head.0.weight"] = torch.zeros(256, 114)
    partial_path = tmp_path / "partial.pt"
    torch.save(partial, partial_path)
    with pytest.raises(RuntimeError, match="partial|Incomplete|both"):
        _runner(tmp_path).load_amp_checkpoint(partial_path)
