from types import SimpleNamespace

import torch

from tracking.mdp.curriculums import parallelism_velocity_curriculum
from tracking.mdp.observations import (
    parallelism_ref_joint_pos_rel_t,
    parallelism_ref_joint_vel_t,
    parallelism_ref_root_pos_b_t,
    parallelism_ref_root_rot_b_t,
)
from tracking.mdp.rewards import (
    parallelism_tracking_errors,
    parallelism_tracking_episode_errors,
    reference_active_swing_foot_max_reward,
    reference_foot_pos_reward,
    reference_joint_max_reward,
    reference_joint_pos_reward,
    reference_root_pos_reward,
    reference_root_rot_reward,
    reset_parallelism_tracking_error_stats,
)
from tracking.mdp.terminations import parallelism_ref_joint_pos_too_far


class _Scene(dict):
    pass


def _fake_env():
    manager = SimpleNamespace(
        current_joint_pos=torch.zeros(2, 12),
        next_joint_pos=torch.zeros(2, 12),
        step_joint_pos=torch.zeros(2, 12),
        current_joint_vel=torch.ones(2, 12),
        step_joint_vel=torch.ones(2, 12),
        current_root_pos_b_policy=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0]]),
        current_root_rot_b_policy=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.2, 0.0]]),
        current_foot_pos_w=torch.tensor(
            [
                [[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]],
                [[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]],
            ]
        ),
        step_foot_pos_w=torch.tensor(
            [
                [[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]],
                [[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]],
            ]
        ),
        step_root_pos_w=torch.zeros(2, 3),
        step_root_rpy_w=torch.zeros(2, 3),
        current_contact_state=torch.ones(2, 4, dtype=torch.bool),
    )
    robot = SimpleNamespace(
        joint_names=(
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ),
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 12),
            joint_vel=torch.zeros(2, 12),
            default_joint_pos=torch.zeros(2, 12),
            root_pos_w=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            body_pos_w=torch.tensor(
                [
                    [[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]],
                    [[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]],
                ]
            ),
        )
    )
    robot.find_bodies = lambda _pattern: ([0, 1, 2, 3], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    env = SimpleNamespace(num_envs=2, device="cpu", parallelism_reference_manager=manager, scene=_Scene(robot=robot))
    return env


def test_reference_observation_shapes():
    env = _fake_env()
    assert parallelism_ref_joint_pos_rel_t(env).shape == (2, 12)
    assert parallelism_ref_joint_vel_t(env).shape == (2, 12)


def test_reference_position_observation_uses_next_frame():
    env = _fake_env()
    env.parallelism_reference_manager.next_joint_pos[:] = 0.25
    assert torch.allclose(parallelism_ref_joint_pos_rel_t(env), torch.full((2, 12), 0.25))


def test_reference_root_pose_observations_use_policy_frame():
    env = _fake_env()
    pos_obs = parallelism_ref_root_pos_b_t(env)
    rot_obs = parallelism_ref_root_rot_b_t(env)
    assert torch.equal(pos_obs, env.parallelism_reference_manager.current_root_pos_b_policy)
    assert torch.equal(rot_obs, env.parallelism_reference_manager.current_root_rot_b_policy)


def test_joint_reward_is_one_when_error_is_zero():
    env = _fake_env()
    reward = reference_joint_pos_reward(env)
    assert torch.allclose(reward, torch.ones(2))


def test_joint_reward_uses_step_target_not_current_reference():
    env = _fake_env()
    env.parallelism_reference_manager.step_joint_pos[:] = 0.1

    reward = reference_joint_pos_reward(env, std=0.5)

    expected = torch.exp(torch.tensor(-(12.0 * 0.1**2) / (0.5**2)))
    assert torch.allclose(reward, torch.full((2,), expected))


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


def test_reference_root_pose_rewards_track_relative_reference():
    env = _fake_env()
    pos_reward = reference_root_pos_reward(env)
    rot_reward = reference_root_rot_reward(env)
    assert torch.allclose(pos_reward, torch.tensor([torch.exp(torch.tensor(-0.2**2 / 0.12**2)), torch.exp(torch.tensor(-0.3**2 / 0.12**2))]))
    assert torch.allclose(rot_reward, torch.tensor([torch.exp(torch.tensor(-0.1**2 / 0.30**2)), torch.exp(torch.tensor(-0.2**2 / 0.30**2))]))


def test_reference_foot_position_reward_is_one_when_feet_match():
    env = _fake_env()
    env.scene["robot"].data.body_pos_w = torch.cat(
        (
            torch.zeros(2, 15, 3),
            env.scene["robot"].data.body_pos_w,
        ),
        dim=1,
    )
    env.scene["robot"].find_bodies = lambda _pattern: (
        [15, 16, 17, 18],
        ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    )

    reward = reference_foot_pos_reward(env)

    assert torch.allclose(reward, torch.ones(2))


def test_active_swing_worst_foot_reward_uses_only_worst_active_leg():
    env = _fake_env()
    env.parallelism_reference_manager.current_contact_state[:] = torch.tensor([False, True, True, False])
    env.scene["robot"].data.body_pos_w[:, 0, 0] += 0.05
    env.scene["robot"].data.body_pos_w[:, 3, 0] += 0.10

    reward = reference_active_swing_foot_max_reward(env, std=0.12)

    expected = torch.exp(torch.tensor(-(0.10 / 0.12) ** 2))
    assert torch.allclose(reward, torch.full((2,), expected))


def test_active_swing_worst_foot_reward_ignores_stance_error():
    env = _fake_env()
    env.parallelism_reference_manager.current_contact_state[:] = torch.tensor([False, True, True, False])
    env.scene["robot"].data.body_pos_w[:, 1, 0] += 1.0

    reward = reference_active_swing_foot_max_reward(env)

    assert torch.allclose(reward, torch.ones(2))


def test_active_swing_worst_foot_reward_is_zero_without_swing_legs():
    env = _fake_env()

    reward = reference_active_swing_foot_max_reward(env)

    assert torch.allclose(reward, torch.zeros(2))


def test_joint_max_reward_uses_worst_of_all_twelve_joints():
    env = _fake_env()
    env.scene["robot"].data.joint_pos[:, 7] = 0.8

    reward = reference_joint_max_reward(env, std=0.8)

    assert torch.allclose(reward, torch.full((2,), torch.exp(torch.tensor(-1.0))))


def test_joint_pos_too_far_triggers_on_max_joint_error():
    env = _fake_env()
    env.scene["robot"].data.joint_pos[1, 3] = 0.9
    done = parallelism_ref_joint_pos_too_far(env, threshold=0.8)
    assert done.tolist() == [False, True]


def test_tracking_errors_use_episode_joint_mean_max_and_reference_pose():
    env = _fake_env()
    env.scene["robot"].data.joint_pos[0, 0] = 0.1
    env.scene["robot"].data.joint_pos[1, 0] = 0.6

    errors = parallelism_tracking_errors(env)

    assert torch.allclose(errors["root_pos_error"], torch.tensor([0.2, 0.3]))
    assert errors["joint_mean_error"][0] < 0.2
    assert errors["joint_max_error"][1] > 0.45


def test_tracking_errors_report_relative_root_pose_metrics():
    env = _fake_env()
    errors = parallelism_tracking_errors(env)
    assert torch.allclose(errors["root_pos_error"], torch.tensor([0.2, 0.3]))
    assert torch.allclose(errors["root_rot_error"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(errors["episode_root_pos_error"], torch.tensor([0.2, 0.3]))
    assert torch.allclose(errors["episode_root_rot_error"], torch.tensor([0.1, 0.2]))


def test_tracking_errors_restore_canonical_foot_order_from_articulation_names():
    env = _fake_env()
    robot = env.scene["robot"]
    shuffled_names = ["RR_foot", "misc", "FL_foot", "RL_foot", "FR_foot"]
    body_pos = torch.zeros(2, len(shuffled_names), 3)
    body_pos[:, 0, 0] = 0.4
    body_pos[:, 2, 0] = 0.1
    body_pos[:, 3, 0] = 0.3
    body_pos[:, 4, 0] = 0.2
    robot.data.body_pos_w = body_pos
    robot.find_bodies = lambda _pattern: ([0, 2, 3, 4], ["RR_foot", "FL_foot", "RL_foot", "FR_foot"])
    env.parallelism_reference_manager.step_foot_pos_w.zero_()

    errors = parallelism_tracking_errors(env)

    assert torch.allclose(errors["foot_error_per_leg"], torch.tensor([[0.1, 0.2, 0.3, 0.4]]).expand(2, -1))


def test_episode_tracking_stats_accumulate_active_swing_and_per_leg_errors():
    env = _fake_env()
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 1
    env.parallelism_reference_manager.current_contact_state[:] = torch.tensor([False, True, True, False])
    env.scene["robot"].data.body_pos_w[:, 0, 0] += 0.1
    env.scene["robot"].data.body_pos_w[:, 3, 2] += 0.2
    env.scene["robot"].data.joint_pos[:, 5] = 0.4
    parallelism_tracking_errors(env)

    env.episode_length_buf += 1
    env.common_step_counter = 2
    env.scene["robot"].data.body_pos_w[:, 0, 0] += 0.2
    env.scene["robot"].data.body_pos_w[:, 3, 2] -= 0.1
    env.scene["robot"].data.joint_pos[:, 8] = 0.6
    stats = parallelism_tracking_errors(env)

    assert torch.allclose(stats["episode_active_swing_foot_mean_error"], torch.full((2,), 0.175))
    assert torch.allclose(stats["episode_active_swing_foot_max_error"], torch.full((2,), 0.3))
    assert torch.allclose(stats["episode_active_swing_foot_z_mean_error"], torch.full((2,), 0.075))
    assert torch.allclose(stats["episode_active_swing_foot_z_max_error"], torch.full((2,), 0.2))
    assert torch.allclose(
        stats["episode_swing_foot_mean_error_per_leg"],
        torch.tensor([[0.2, 0.0, 0.0, 0.15]]).expand(2, -1),
    )
    assert torch.allclose(
        stats["episode_swing_foot_max_error_per_leg"],
        torch.tensor([[0.3, 0.0, 0.0, 0.2]]).expand(2, -1),
    )
    assert torch.allclose(
        stats["episode_swing_foot_z_mean_error_per_leg"],
        torch.tensor([[0.0, 0.0, 0.0, 0.15]]).expand(2, -1),
    )
    assert torch.allclose(
        stats["episode_joint_max_error_per_leg"],
        torch.tensor([[0.0, 0.4, 0.6, 0.0]]).expand(2, -1),
    )


def test_reset_tracking_stats_clears_only_selected_environment_per_leg_buffers():
    env = _fake_env()
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.common_step_counter = 1
    env.parallelism_reference_manager.current_contact_state[:] = torch.tensor([False, True, True, False])
    env.scene["robot"].data.body_pos_w[:, 0, 0] += 0.1
    parallelism_tracking_errors(env)

    reset_parallelism_tracking_error_stats(env, torch.tensor([1]))
    stats = parallelism_tracking_episode_errors(env)

    assert stats["episode_active_swing_foot_mean_error"][0] > 0.0
    assert stats["episode_active_swing_foot_mean_error"][1] == 0.0
    assert stats["episode_swing_foot_max_error_per_leg"][0, 0] > 0.0
    assert stats["episode_swing_foot_max_error_per_leg"][1, 0] == 0.0


def test_tracking_error_cache_reuses_current_step_errors():
    env = _fake_env()
    env.common_step_counter = 10
    env.scene["robot"].data.joint_pos[0, 0] = 0.1
    first = parallelism_tracking_errors(env)

    env.scene["robot"].data.joint_pos[0, 0] = 0.7
    second = parallelism_tracking_errors(env)

    assert torch.allclose(first["joint_max_error"], second["joint_max_error"])

    env.common_step_counter = 11
    third = parallelism_tracking_errors(env)

    assert third["joint_max_error"][0] > second["joint_max_error"][0]


def test_velocity_curriculum_blocks_upgrade_when_episode_joint_max_is_high():
    env = _fake_env()
    env.reset_time_outs = torch.ones(2, dtype=torch.bool)
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    env._parallelism_tracking_error_frames = torch.ones(2)
    env._parallelism_tracking_joint_mean_sum = torch.tensor([0.01, 0.01])
    env._parallelism_tracking_joint_max = torch.tensor([0.1, 0.7])
    env._parallelism_tracking_root_pos_sum = torch.tensor([0.01, 0.02])
    env._parallelism_tracking_root_rot_sum = torch.tensor([0.01, 0.02])
    ranges = SimpleNamespace(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.05, 0.05), ang_vel_z=(-0.2, 0.2))
    limit_ranges = SimpleNamespace(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0))
    env.command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(ranges=ranges, limit_ranges=limit_ranges)))

    level = parallelism_velocity_curriculum(
        env,
        torch.tensor([0, 1]),
        root_pos_threshold=0.12,
        root_rot_threshold=0.30,
        joint_mean_threshold=0.2,
        joint_max_threshold=0.45,
    )

    assert level.item() == 0.5
