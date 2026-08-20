from pathlib import Path
from types import SimpleNamespace

import torch

from tracking.mdp.observations import (
    parallelism_plan_valid,
    parallelism_ref_joint_pos_rel_t,
    parallelism_ref_joint_vel_t,
    parallelism_ref_root_pos_b_t,
    parallelism_ref_root_rot_b_t,
)
from tracking.mdp.rewards import (
    parallelism_tracking_errors,
    reference_joint_pos_reward,
    reference_root_pos_reward,
)
from tracking.mdp.terminations import parallelism_ref_joint_pos_too_far


class _Scene(dict):
    pass


def _env():
    manager = SimpleNamespace(
        step_plan_valid=torch.tensor([True, False]),
        plan_valid=torch.tensor([True, False]),
        current_joint_pos=torch.zeros(2, 12),
        next_joint_pos=torch.full((2, 12), 0.25),
        step_joint_pos=torch.zeros(2, 12),
        current_joint_vel=torch.ones(2, 12),
        step_joint_vel=torch.ones(2, 12),
        current_root_pos_b_policy=torch.tensor([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        current_root_rot_b_policy=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]]),
        step_root_pos_w=torch.zeros(2, 3),
        step_root_rpy_w=torch.zeros(2, 3),
        step_foot_pos_w=torch.zeros(2, 4, 3),
        current_contact_state=torch.ones(2, 4, dtype=torch.bool),
        standstill_latched=torch.tensor([False, True]),
        standstill_count=torch.zeros(2, dtype=torch.long),
    )
    robot = SimpleNamespace(
        joint_names=tuple(
            name
            for leg in ("FL", "FR", "RL", "RR")
            for name in (f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint")
        ),
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 12),
            joint_vel=torch.zeros(2, 12),
            default_joint_pos=torch.zeros(2, 12),
            root_pos_w=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            body_pos_w=torch.zeros(2, 4, 3),
        ),
    )
    robot.find_bodies = lambda _pattern: ([0, 1, 2, 3], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        common_step_counter=1,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        parallelism_reference_manager=manager,
        scene=_Scene(robot=robot),
    )
    return env


def test_plan_valid_observation_is_explicit_and_reference_is_zeroed_for_invalid_env():
    env = _env()

    assert torch.equal(parallelism_plan_valid(env), torch.tensor([[1.0], [0.0]]))
    assert torch.allclose(parallelism_ref_joint_pos_rel_t(env)[0], torch.full((12,), 0.25))
    assert torch.allclose(parallelism_ref_joint_pos_rel_t(env)[1], torch.zeros(12))
    assert torch.allclose(parallelism_ref_joint_vel_t(env)[1], torch.zeros(12))
    assert torch.allclose(parallelism_ref_root_pos_b_t(env)[1], torch.zeros(3))
    assert torch.allclose(parallelism_ref_root_rot_b_t(env)[1], torch.zeros(3))


def test_reference_rewards_are_disabled_only_for_invalid_plans():
    env = _env()
    assert reference_joint_pos_reward(env).tolist()[0] > 0.0
    assert reference_joint_pos_reward(env).tolist()[1] == 0.0
    assert reference_root_pos_reward(env).tolist()[0] > 0.0
    assert reference_root_pos_reward(env).tolist()[1] == 0.0


def test_reference_joint_termination_is_masked_for_invalid_plans():
    env = _env()
    env.scene["robot"].data.joint_pos[1, 0] = 1.0
    assert parallelism_ref_joint_pos_too_far(env, threshold=0.8, consecutive_steps=1).tolist() == [False, False]

    env.parallelism_reference_manager.step_plan_valid[:] = True
    assert parallelism_ref_joint_pos_too_far(env, threshold=0.8, consecutive_steps=1).tolist() == [False, True]


def test_tracking_metrics_keep_valid_and_invalid_frames_separate():
    env = _env()
    env.scene["robot"].data.joint_pos[0, 0] = 0.1
    env.scene["robot"].data.joint_pos[1, 0] = 0.7
    stats = parallelism_tracking_errors(env)

    assert stats["valid_episode_reference_frame_count"].tolist() == [1.0, 0.0]
    assert stats["invalid_episode_reference_frame_count"].tolist() == [0.0, 1.0]
    assert stats["valid_episode_joint_max_error"][0] == 0.1
    assert stats["invalid_episode_joint_max_error"][1] == 0.7


def test_large_teacher_overrides_collision_swing_reward_and_command_period():
    source = Path(__file__).resolve().parents[2] / "tracking/parallelism_cross_large_complex_env_cfg.py"
    text = source.read_text()
    assert "parallelism_geometry_collision = RewTerm(" in text
    assert "weight=-10.0" in text
    assert "active_swing_foot_on_small_obstacle = RewTerm(" in text
    assert "weight=10.0" in text
    assert "resampling_time_range = (10.0, 10.0)" in text
