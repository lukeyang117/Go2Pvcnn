from types import SimpleNamespace

import torch

from tracking.mdp.curriculums import parallelism_velocity_curriculum
from tracking.mdp.observations import (
    parallelism_ref_joint_pos_rel_t,
    parallelism_ref_joint_vel_t,
    parallelism_ref_root_lin_vel_b_t,
)
from tracking.mdp.rewards import (
    parallelism_tracking_errors,
    reference_joint_pos_reward,
    reference_root_lin_vel_reward,
)
from tracking.mdp.terminations import parallelism_ref_joint_pos_too_far


class _Scene(dict):
    pass


def _fake_env():
    manager = SimpleNamespace(
        current_joint_pos=torch.zeros(2, 12),
        current_joint_vel=torch.ones(2, 12),
        current_root_lin_vel_b_policy=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0]]),
        current_root_ang_vel_b_policy=torch.zeros(2, 3),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 12),
            joint_vel=torch.zeros(2, 12),
            default_joint_pos=torch.zeros(2, 12),
            root_lin_vel_b=torch.zeros(2, 3),
            root_ang_vel_b=torch.zeros(2, 3),
        )
    )
    env = SimpleNamespace(num_envs=2, device="cpu", parallelism_reference_manager=manager, scene=_Scene(robot=robot))
    return env


def test_reference_observation_shapes():
    env = _fake_env()
    assert parallelism_ref_joint_pos_rel_t(env).shape == (2, 12)
    assert parallelism_ref_joint_vel_t(env).shape == (2, 12)


def test_reference_root_velocity_observation_uses_policy_frame():
    env = _fake_env()
    obs = parallelism_ref_root_lin_vel_b_t(env)
    assert torch.equal(obs, env.parallelism_reference_manager.current_root_lin_vel_b_policy)


def test_joint_reward_is_one_when_error_is_zero():
    env = _fake_env()
    reward = reference_joint_pos_reward(env)
    assert torch.allclose(reward, torch.ones(2))


def test_joint_reward_uses_instinctlab_sum_square_gaussian():
    env = _fake_env()
    env.scene["robot"].data.joint_pos[0, 0] = 0.1
    env.scene["robot"].data.joint_pos[0, 1] = -0.2

    reward = reference_joint_pos_reward(env, std=0.5)

    expected = torch.exp(torch.tensor(-(0.1**2 + 0.2**2) / (0.5**2)))
    assert torch.allclose(reward[0], expected)


def test_joint_reward_tolerance_ignores_small_errors():
    env = _fake_env()
    env.scene["robot"].data.joint_pos[0, 0] = 0.04
    env.scene["robot"].data.joint_pos[0, 1] = 0.08

    reward = reference_joint_pos_reward(env, std=0.5, tracking_tolerance=0.05)

    expected = torch.exp(torch.tensor(-((0.08 - 0.05) ** 2) / (0.5**2)))
    assert torch.allclose(reward[0], expected)


def test_reference_root_velocity_reward_tracks_reference_not_command():
    env = _fake_env()
    env.scene["robot"].data.root_lin_vel_b[:] = env.parallelism_reference_manager.current_root_lin_vel_b_policy
    env.command_manager = SimpleNamespace(get_command=lambda name: torch.ones(2, 3) * 9.0)

    reward = reference_root_lin_vel_reward(env)

    assert torch.allclose(reward, torch.ones(2))


def test_joint_pos_too_far_triggers_on_max_joint_error():
    env = _fake_env()
    env.scene["robot"].data.joint_pos[1, 3] = 0.9
    done = parallelism_ref_joint_pos_too_far(env, threshold=0.8)
    assert done.tolist() == [False, True]


def test_tracking_errors_use_episode_joint_mean_max_and_reference_velocity():
    env = _fake_env()
    env.scene["robot"].data.root_lin_vel_b[:] = env.parallelism_reference_manager.current_root_lin_vel_b_policy
    env.scene["robot"].data.joint_pos[0, 0] = 0.1
    env.scene["robot"].data.joint_pos[1, 0] = 0.6

    errors = parallelism_tracking_errors(env)

    assert torch.allclose(errors["lin_vel_error"], torch.zeros(2))
    assert errors["joint_mean_error"][0] < 0.2
    assert errors["joint_max_error"][1] > 0.45


def test_velocity_curriculum_blocks_upgrade_when_episode_joint_max_is_high():
    env = _fake_env()
    env.reset_time_outs = torch.ones(2, dtype=torch.bool)
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    env._parallelism_tracking_error_frames = torch.ones(2)
    env._parallelism_tracking_joint_mean_sum = torch.tensor([0.01, 0.01])
    env._parallelism_tracking_joint_max = torch.tensor([0.1, 0.7])
    env._parallelism_tracking_lin_vel_sum = torch.zeros(2)
    env._parallelism_tracking_ang_vel_sum = torch.zeros(2)
    ranges = SimpleNamespace(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.2, 0.2))
    limit_ranges = SimpleNamespace(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))
    env.command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(ranges=ranges, limit_ranges=limit_ranges)))

    level = parallelism_velocity_curriculum(
        env,
        torch.tensor([0, 1]),
        joint_mean_threshold=0.2,
        joint_max_threshold=0.45,
    )

    assert level.item() == 0.5
