from __future__ import annotations

import torch


def test_active_swing_foot_reward_requires_small_obstacle_and_no_collision() -> None:
    from tracking.mdp.rewards import _active_small_obstacle_safe_mask

    semantic = torch.tensor([[1, 1, 0, 1]])
    height = torch.tensor([[0.16, 0.16, 0.0, 0.16]])
    foot_z = torch.tensor([[0.16, 0.22, 0.16, 0.16]])
    valid = torch.ones_like(semantic, dtype=torch.bool)
    collision = torch.tensor([[False, False, False, True]])
    contact = torch.tensor([[False, False, True, False]])

    safe = _active_small_obstacle_safe_mask(
        semantic,
        height,
        foot_z,
        valid,
        collision,
        contact,
        touchdown_tolerance_m=0.04,
    )

    assert safe.tolist() == [[True, False, False, False]]


def test_collision_penalty_is_one_for_active_collision() -> None:
    from tracking.mdp.rewards import _active_collision_penalty

    collision = torch.tensor([[True, True, False, False]])
    contact = torch.tensor([[False, True, True, False]])

    penalty = _active_collision_penalty(collision, contact)

    assert penalty.tolist() == [1.0]
