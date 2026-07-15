"""Named weighted objective assembly."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from extension.joint_mpc_rti.losses.barriers import relaxed_barrier


def weighted_objective(losses: Mapping[str, Tensor], weights: Mapping[str, float]) -> Tensor:
    if not losses:
        raise ValueError("losses must not be empty")
    missing = set(losses) - set(weights)
    if missing:
        raise KeyError(f"missing weights for loss terms: {sorted(missing)}")
    first = next(iter(losses.values()))
    total = torch.zeros_like(first)
    for name, value in losses.items():
        total = total + float(weights[name]) * torch.as_tensor(value, dtype=first.dtype, device=first.device)
    return total


def terminal_losses(
    *,
    terminal_control: Tensor,
    command_body: Tensor,
    terminal_root_rpy: Tensor,
    terminal_joint_pos: Tensor,
    nominal_joint_pos: Tensor,
    obstacle_distance: Tensor,
    obstacle_approach_speed: Tensor,
    contact_viability: Tensor,
    obstacle_margin: float = 0.16,
    barrier_relaxation: float = 0.01,
) -> dict[str, Tensor]:
    control = torch.as_tensor(terminal_control)
    command = torch.as_tensor(command_body, dtype=control.dtype, device=control.device)
    root_rpy = torch.as_tensor(terminal_root_rpy, dtype=control.dtype, device=control.device)
    joint = torch.as_tensor(terminal_joint_pos, dtype=control.dtype, device=control.device)
    nominal_joint = torch.as_tensor(nominal_joint_pos, dtype=control.dtype, device=control.device)
    distance = torch.as_tensor(obstacle_distance, dtype=control.dtype, device=control.device)
    approach = torch.as_tensor(obstacle_approach_speed, dtype=control.dtype, device=control.device)
    viability = torch.as_tensor(contact_viability, dtype=control.dtype, device=control.device)
    terminal_command = torch.stack((control[:, 0], control[:, 1], control[:, 5]), dim=-1)
    command_loss = ((terminal_command - command) ** 2).mean(dim=1)
    obstacle_loss = relaxed_barrier(
        distance - float(obstacle_margin), relaxation=barrier_relaxation
    ) + torch.relu(-approach) ** 2
    posture_loss = (root_rpy[:, :2] ** 2).mean(dim=1) + ((joint - nominal_joint) ** 2).mean(dim=1)
    viability_loss = (1.0 - viability.clamp(0.0, 1.0)) ** 2
    return {
        "terminal_command_velocity": command_loss,
        "terminal_obstacle_safety": obstacle_loss,
        "terminal_posture": posture_loss,
        "terminal_contact_viability": viability_loss,
    }


__all__ = ["terminal_losses", "weighted_objective"]
