import torch

from ame_baseline.ame_env_wrapper import AmeRslRlEnvWrapper
from ame_baseline.ame_train_cfg import get_ame_train_cfg


def test_ame_wrapper_flattens_map_before_state():
    wrapper = AmeRslRlEnvWrapper.__new__(AmeRslRlEnvWrapper)
    obs, extras = wrapper._format_observations(
        {
            "policy_elevation_semantic_map": torch.zeros(2, 6, 16, 16),
            "policy_state": torch.ones(2, 45),
            "critic_elevation_semantic_map": torch.zeros(2, 6, 16, 16),
            "critic_state": torch.ones(2, 48),
        }
    )
    assert obs.shape == (2, 6 * 16 * 16 + 45)
    assert extras["observations"]["critic"].shape == (2, 6 * 16 * 16 + 48)


def test_ame_cfg_matches_ppo_baseline():
    cfg = get_ame_train_cfg()
    assert cfg["num_steps_per_env"] == 40
    assert cfg["algorithm"]["num_mini_batches"] == 4
    assert cfg["algorithm"]["desired_kl"] == 0.01
