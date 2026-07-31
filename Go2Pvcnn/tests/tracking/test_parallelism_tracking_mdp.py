from types import SimpleNamespace

import torch

from tracking.mdp.observations import parallelism_ref_joint_pos_rel_t, parallelism_ref_joint_vel_t
from tracking.mdp.rewards import reference_joint_pos_reward
from tracking.mdp.terminations import parallelism_ref_joint_pos_too_far


class _Scene(dict):
    pass


def _fake_env():
    manager = SimpleNamespace(
        current_joint_pos=torch.zeros(2, 12),
        current_joint_vel=torch.ones(2, 12),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 12),
            joint_vel=torch.zeros(2, 12),
            default_joint_pos=torch.zeros(2, 12),
        )
    )
    env = SimpleNamespace(parallelism_reference_manager=manager, scene=_Scene(robot=robot))
    return env


def test_reference_observation_shapes():
    env = _fake_env()
    assert parallelism_ref_joint_pos_rel_t(env).shape == (2, 12)
    assert parallelism_ref_joint_vel_t(env).shape == (2, 12)


def test_joint_reward_is_one_when_error_is_zero():
    env = _fake_env()
    reward = reference_joint_pos_reward(env)
    assert torch.allclose(reward, torch.ones(2))


def test_joint_pos_too_far_triggers_on_max_joint_error():
    env = _fake_env()
    env.scene["robot"].data.joint_pos[1, 3] = 0.9
    done = parallelism_ref_joint_pos_too_far(env, threshold=0.8)
    assert done.tolist() == [False, True]
