import torch

from m1_ame.m1_mixed_action_math import split_m1_mixed_action


M1_USD_JOINT_NAMES = (
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT", "FBL_FOOT_JOINT",
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT", "FAR_FOOT_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT", "RBL_FOOT_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT", "RAR_FOOT_JOINT",
)


def test_m1_mixed_action_preserves_usd_order_and_separates_control_modes():
    actions = torch.zeros((2, 16))
    actions[:, 0] = 1.0
    actions[:, 3] = 0.5
    actions[:, 15] = -0.25
    leg_indices = tuple(i for i, name in enumerate(M1_USD_JOINT_NAMES) if not name.endswith("FOOT_JOINT"))
    wheel_indices = tuple(i for i, name in enumerate(M1_USD_JOINT_NAMES) if name.endswith("FOOT_JOINT"))
    default_leg_pos = torch.arange(12, dtype=torch.float32).view(1, 12).repeat(2, 1)

    position_targets, velocity_targets = split_m1_mixed_action(
        actions, leg_indices, wheel_indices, 0.25, 8.0, default_leg_pos
    )

    assert position_targets.shape == (2, 12)
    assert velocity_targets.shape == (2, 4)
    assert torch.allclose(position_targets[:, 0], default_leg_pos[:, 0] + 0.25)
    assert torch.allclose(velocity_targets[:, 0], torch.full((2,), 4.0))
    assert torch.allclose(velocity_targets[:, 3], torch.full((2,), -2.0))


def test_m1_mixed_action_source_declares_a_single_sixteen_dimensional_term():
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[2] / "m1_ame" / "m1_mixed_actions.py"
    ).read_text(encoding="utf-8")
    assert "class M1MixedJointActionCfg" in source
    assert "class M1MixedJointAction" in source
    assert "set_joint_position_target" in source
    assert "set_joint_velocity_target" in source
