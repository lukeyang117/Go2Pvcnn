"""Adjacent-state body-command tracking residuals."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg


def command_residual(state: Tensor, command_body: Tensor, schedule, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    command = torch.as_tensor(command_body, dtype=trajectory.dtype, device=trajectory.device)
    delta_xy = (trajectory[:, 1:, :2] - trajectory[:, :-1, :2]) / float(cfg.runtime.dt)
    yaw = trajectory[:, :-1, 5]
    body_velocity = torch.stack(
        (
            torch.cos(yaw) * delta_xy[..., 0] + torch.sin(yaw) * delta_xy[..., 1],
            -torch.sin(yaw) * delta_xy[..., 0] + torch.cos(yaw) * delta_xy[..., 1],
        ),
        dim=-1,
    )
    yaw_rate = (trajectory[:, 1:, 5] - trajectory[:, :-1, 5]) / float(cfg.runtime.dt)
    linear = math.sqrt(float(cfg.loss_terms.command_linear)) * (body_velocity - command[:, None, :2])
    angular = math.sqrt(float(cfg.loss_terms.command_yaw)) * (yaw_rate - command[:, None, 2])
    residual = torch.cat((linear, angular[..., None]), dim=-1)
    continuing_swing = torch.logical_and(schedule.swing[:, 1:], schedule.swing[:, :-1])
    early_phase = (
        (1.0 - schedule.swing_tau[:, :1]) * continuing_swing[:, :1]
    ).amax(dim=-1).to(trajectory.dtype)
    command_norm_sq = command.square().sum(dim=-1)
    activity_scale_sq = float(cfg.loss_terms.command_activity_scale) ** 2
    command_activity = -torch.expm1(-command_norm_sq[:, None] / activity_scale_sq)
    early_weight = float(cfg.loss_terms.command_early_swing)
    first_edge_weight = 1.0 - (1.0 - early_weight) * early_phase * command_activity
    phase_weight = torch.cat((first_edge_weight, torch.ones_like(residual[:, 1:, 0])), dim=1)
    residual = residual * phase_weight.clamp_min(0.0).sqrt()[..., None]
    return residual.flatten(1) / math.sqrt(float(residual[0].numel()))


def command_loss(state: Tensor, command_body: Tensor, schedule, cfg: JointMpcRtiCfg) -> Tensor:
    residual = command_residual(state, command_body, schedule, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["command_loss", "command_residual"]
