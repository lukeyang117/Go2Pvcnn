from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "m1_ame"


def test_m1_reward_module_declares_explicit_joint_and_body_selectors():
    source = (PACKAGE / "m1_ame_rewards.py").read_text(encoding="utf-8")

    assert "M1_PLANNER_JOINT_NAMES" in source
    assert "M1_WHEEL_JOINT_NAMES" in source
    assert "M1_SUPPORT_BODY_NAMES" in source
    assert "M1_WHEEL_RADIUS_M" in source
    assert "M1_WHEEL_THICKNESS_M" in source
    assert "M1_WHEEL_HORIZONTAL_ENVELOPE_M" in source
    assert 'joint_names=".*"' not in source
    assert 'body_names=".*_foot"' not in source


def test_m1_reward_config_has_no_go2_reference_rewards_and_has_m1_geometry_term():
    source = (PACKAGE / "m1_ame_rewards.py").read_text(encoding="utf-8")

    assert "class M1AmeRewardsCfg" in source
    assert "class M1AmeTerminationsCfg" in source
    assert "reference_joint_pos" not in source
    assert "reference_foot_pos" not in source
    assert "reference_root_pos" not in source
    assert "m1_geometry_collision_penalty" in source
    assert "M1_ROOT_Z_M" in source


def test_m1_environment_uses_m1_reward_and_termination_configs():
    source = (PACKAGE / "m1_ame_env_cfg.py").read_text(encoding="utf-8")

    assert "M1AmeRewardsCfg" in source
    assert "M1AmeTerminationsCfg" in source
