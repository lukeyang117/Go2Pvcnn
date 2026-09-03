from __future__ import annotations

import torch

from evaluation.benchmark_suites import goal_command


def test_goal_command_has_zero_lateral_velocity() -> None:
    command = goal_command(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([[1.0, 0.0]]),
        0.5,
    )
    assert command.shape == (1, 3)
    assert command[0, 0].item() == 0.5
    assert command[0, 1].item() == 0.0
    assert command[0, 2].item() == 0.0


def test_goal_command_wraps_heading_error() -> None:
    command = goal_command(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([3.1]),
        torch.tensor([[-1.0, 0.0]]),
        1.0,
    )
    assert -1.0 <= command[0, 2].item() <= 1.0
