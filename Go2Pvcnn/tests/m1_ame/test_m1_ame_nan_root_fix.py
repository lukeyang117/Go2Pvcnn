from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))

from m1_ame.m1_actor_critic_ame import M1ActorCriticAME
from m1_ame.m1_mixed_action_math import split_m1_mixed_action


def _small_policy() -> M1ActorCriticAME:
    return M1ActorCriticAME(
        num_actor_obs=6 * 4 * 4 + 2,
        num_critic_obs=6 * 4 * 4 + 3,
        num_actions=16,
        map_channels=6,
        map_size=4,
        mha_dim=4,
        num_heads=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        init_noise_std=[0.2] * 12 + [0.1] * 4,
    )


def test_m1_normalized_actions_are_bounded_and_leg_targets_are_slew_limited():
    actions = torch.zeros((1, 16))
    actions[0, 0] = 100.0
    default_leg = torch.zeros((1, 12))
    previous_target = torch.zeros((1, 12))
    lo = torch.full((12,), -2.5)
    hi = torch.full((12,), 2.5)

    leg_target, wheel_target = split_m1_mixed_action(
        actions,
        leg_indices=(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14),
        wheel_indices=(3, 7, 11, 15),
        leg_scale=0.15,
        wheel_scale=8.0,
        default_leg_pos=default_leg,
        wheel_velocity_limit=8.0,
        leg_position_lower=lo,
        leg_position_upper=hi,
        normalized_action_limit=1.0,
        previous_leg_pos_target=previous_target,
        leg_target_delta_limit=0.10,
    )

    torch.testing.assert_close(leg_target[0, 0], torch.tensor(0.10), atol=1e-6, rtol=0)
    assert torch.max(torch.abs(leg_target - previous_target)).item() <= 0.100001
    assert torch.equal(wheel_target, torch.zeros_like(wheel_target))


def test_m1_policy_uses_squashed_actions_and_group_specific_std_caps():
    policy = _small_policy()
    for layer in policy.actor.modules():
        if isinstance(layer, nn.Linear):
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()
    with torch.no_grad():
        policy.actor[-1].bias.fill_(8.0)
        policy.std.fill_(0.8)
    policy.clip_std()

    obs = torch.zeros((3, 6 * 4 * 4 + 2))
    actions = policy.act(obs)
    log_prob = policy.get_actions_log_prob(actions)

    assert torch.isfinite(actions).all()
    assert torch.all(actions.abs() <= 1.0)
    assert torch.isfinite(log_prob).all()
    wheel_indices = torch.tensor((3, 7, 11, 15))
    leg_indices = torch.tensor((0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14))
    assert torch.all(policy.std[leg_indices] <= 0.30)
    assert torch.all(policy.std[wheel_indices] <= 0.10)


def test_m1_joint_position_reward_uses_leg_selector_and_not_all_joints():
    source = (ROOT / "m1_ame" / "m1_ame_rewards.py").read_text(encoding="utf-8")

    assert "func=m1_joint_position_penalty" in source
    assert "func=m1_joint_vel_l2" in source
    assert "asset.data.joint_pos[:, ids]" in source
    assert "asset.data.joint_vel[:, ids]" in source


def test_m1_asset_separates_leg_and_wheel_actuator_limits():
    source = (ROOT / "go2_pvcnn" / "assets" / "m1.py").read_text(encoding="utf-8")

    assert '"leg_joints": ImplicitActuatorCfg(' in source
    assert '"wheel_joints": ImplicitActuatorCfg(' in source
    assert "effort_limit_sim=150.0" in source
    assert "velocity_limit_sim=20.0" in source
    assert "effort_limit_sim=50.0" in source
    assert "velocity_limit_sim=8.0" in source
    assert "stiffness=0.0" in source


def test_m1_invalid_state_detector_catches_nonfinite_and_overspeed():
    from m1_ame.m1_ame_safety import m1_invalid_state_mask

    data = type("Data", (), {})()
    data.joint_pos = torch.zeros((3, 4))
    data.joint_vel = torch.zeros((3, 4))
    data.joint_acc = torch.zeros((3, 4))
    data.applied_torque = torch.zeros((3, 4))
    data.root_pos_w = torch.zeros((3, 3))
    data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    data.root_lin_vel_w = torch.zeros((3, 3))
    data.root_ang_vel_w = torch.zeros((3, 3))
    data.body_lin_vel_w = torch.zeros((3, 5, 3))
    data.joint_vel[1, 0] = 21.0  # leg speed limit is 20 rad/s
    data.joint_vel[2, 3] = float("inf")

    bad = m1_invalid_state_mask(
        data,
        leg_joint_ids=(0, 1, 2),
        wheel_joint_ids=(3,),
        leg_velocity_limit=20.0,
        wheel_velocity_limit=8.0,
    )

    assert torch.equal(bad, torch.tensor([False, True, True]))


def test_m1_sanitized_state_rewards_remain_finite_for_invalid_state():
    from m1_ame.m1_ame_safety import finite_safe_tensor

    values = torch.tensor([[1.0, float("nan"), float("inf"), -1.0e30]])
    safe = finite_safe_tensor(values, lower=-100.0, upper=100.0)

    assert torch.isfinite(safe).all()
    assert torch.equal(safe, torch.tensor([[1.0, 0.0, 0.0, -100.0]]))


def test_m1_invalid_quaternions_fall_back_to_identity_before_fk():
    from m1_ame.m1_ame_safety import finite_safe_quaternion_wxyz

    q = torch.tensor([[float("nan"), 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    safe = finite_safe_quaternion_wxyz(q)

    assert torch.isfinite(safe).all()
    torch.testing.assert_close(safe[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(safe[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(safe[2], torch.tensor([1.0, 0.0, 0.0, 0.0]))
