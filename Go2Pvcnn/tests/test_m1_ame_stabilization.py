from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))

from m1_ame.m1_mixed_action_math import split_m1_mixed_action


def test_m1_leg_targets_are_clamped_to_robot_limits():
    actions = torch.tensor([[10.0, -10.0, 10.0, -10.0]])
    default = torch.zeros((1, 2))
    lower = torch.tensor([[-0.5, -1.0]])
    upper = torch.tensor([[0.5, 1.0]])

    position_targets, _ = split_m1_mixed_action(
        actions,
        leg_indices=(0, 1),
        wheel_indices=(2, 3),
        leg_scale=0.25,
        wheel_scale=8.0,
        default_leg_pos=default,
        wheel_velocity_limit=8.0,
        leg_position_lower=lower,
        leg_position_upper=upper,
    )

    assert torch.equal(position_targets, torch.tensor([[0.5, -1.0]]))


def test_ame_actor_critic_accepts_per_action_noise_std():
    from m1_ame.m1_actor_critic_ame import M1ActorCriticAME

    policy = M1ActorCriticAME(
        num_actor_obs=6 * 16 * 16 + 4,
        num_critic_obs=6 * 16 * 16 + 4,
        num_actions=4,
        init_noise_std=[0.2, 0.2, 0.15, 0.15],
    )

    assert policy.std.shape == (4,)
    assert torch.allclose(policy.std.detach(), torch.tensor([0.2, 0.2, 0.15, 0.15]))


def test_ppo_kl_helper_can_normalize_by_action_dimension():
    from rsl_rl.algorithms.ppo import approximate_gaussian_kl

    old_mu = torch.zeros((2, 4))
    old_sigma = torch.ones((2, 4))
    mu = torch.full((2, 4), 0.1)
    sigma = torch.ones((2, 4))

    summed = approximate_gaussian_kl(old_mu, old_sigma, mu, sigma, normalize_by_action_dim=False)
    normalized = approximate_gaussian_kl(old_mu, old_sigma, mu, sigma, normalize_by_action_dim=True)

    assert torch.allclose(normalized, summed / 4.0)


def test_m1_gait_terms_and_initial_reset_guard_are_enabled():
    rewards = (ROOT / "m1_ame" / "m1_ame_rewards.py").read_text()
    curriculum = (ROOT / "go2_pvcnn" / "mdp" / "curriculums.py").read_text()

    assert "feet_air_time = RewTerm(" in rewards
    assert "air_time_variance = RewTerm(" in rewards
    assert "feet_slide = RewTerm(" in rewards
    assert "fixed_gait_contact_match = RewTerm(" in rewards
    assert "def _is_initial_reset(env)" in curriculum
    assert "if _is_initial_reset(env):" in curriculum
