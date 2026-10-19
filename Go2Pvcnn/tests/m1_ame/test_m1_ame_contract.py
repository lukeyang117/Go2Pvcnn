from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "m1_ame"


def test_m1_ame_package_exposes_dimension_contract():
    from m1_ame import (
        M1_AME_ACTION_DIM,
        M1_AME_CRITIC_OBS_DIM,
        M1_AME_ENV_ID,
        M1_AME_POLICY_OBS_DIM,
    )

    assert M1_AME_ENV_ID == "Isaac-M1-Cross-Large-Complex-AME-v0"
    assert M1_AME_POLICY_OBS_DIM == 1594
    assert M1_AME_CRITIC_OBS_DIM == 1597
    assert M1_AME_ACTION_DIM == 16


def test_m1_ame_config_contains_m1_asset_and_full_joint_state():
    source = (PACKAGE / "m1_ame_env_cfg.py").read_text(encoding="utf-8")

    assert "M1_CFG" in source
    assert "M1_USD_JOINT_NAMES" in source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/BASE_LINK"' in source
    assert "joint_pos_rel" in source
    assert "joint_vel_rel" in source
    assert "last_action" in source
    assert "planner_owned_reference_cache = False" in source
    assert "use_batched_reference_trajectory = False" in source
    assert "m1_reset_root_state_uniform" in source
    assert "sync_root_to_terrain" in source


def test_m1_ame_config_uses_mixed_leg_position_wheel_velocity_action():
    source = (PACKAGE / "m1_ame_env_cfg.py").read_text(encoding="utf-8")

    assert "M1MixedJointActionCfg" in source
    assert "leg_joint_names=list(M1_LEG_JOINT_NAMES)" in source
    assert "wheel_joint_names=list(M1_WHEEL_JOINT_NAMES)" in source
    assert "wheel_scale=8.0" in source
    assert "JointPositionActionCfg" not in source


def test_m1_ame_does_not_add_parallelism_reference_observations():
    source = (PACKAGE / "m1_ame_env_cfg.py").read_text(encoding="utf-8")

    assert "parallelism_ref_joint_pos" not in source
    assert "parallelism_ref_joint_vel" not in source
    assert "parallelism_ref_root_pos" not in source
    assert "ParallelismReferenceManager" not in source
    assert "trajectory_manager" not in source
