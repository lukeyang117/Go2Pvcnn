from pathlib import Path

def test_ame_config_uses_six_channel_helper_without_noise():
    source = Path("ame_baseline/ame_env_cfg.py").read_text()
    assert "downsampled_ame_scan" in source
    assert "noise=None" in source
    assert "num_envs=1024" in source


def test_ame_observation_groups_have_matching_policy_and_critic_maps():
    source = Path("ame_baseline/ame_env_cfg.py").read_text()
    assert '"target_size": 16' in source
    assert "policy_elevation_semantic_map: MapCfg" in source
    assert "critic_elevation_semantic_map: MapCfg" in source
