"""Swing-foot progress relative to root progress."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk


def swing_speed_penalty(foot_step: Tensor, root_step: Tensor, *, margin: float) -> Tensor:
    return torch.relu(torch.as_tensor(root_step) + float(margin) - torch.as_tensor(foot_step))


def directional_progress(
    step_xy: Tensor,
    command_body: Tensor,
    yaw_w: Tensor,
    *,
    activity_scale: float,
) -> Tensor:
    """Project world-frame displacement onto the continuous body-command axis."""
    step = torch.as_tensor(step_xy)
    command = torch.as_tensor(command_body, dtype=step.dtype, device=step.device)
    yaw = torch.as_tensor(yaw_w, dtype=step.dtype, device=step.device)
    command_xy = command[:, :2]
    norm = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True)
    axis_body = command_xy / norm.clamp_min(float(activity_scale))
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    axis_world = torch.stack(
        (
            cosine * axis_body[:, None, 0] - sine * axis_body[:, None, 1],
            sine * axis_body[:, None, 0] + cosine * axis_body[:, None, 1],
        ),
        dim=-1,
    )
    while axis_world.ndim < step.ndim:
        axis_world = axis_world.unsqueeze(-2)
    activity = 1.0 - torch.exp(-norm.square() / float(activity_scale) ** 2)
    while activity.ndim < step.ndim - 1:
        activity = activity.unsqueeze(1)
    return (step * axis_world).sum(dim=-1) * activity


def swing_speed_residual(
    state: Tensor,
    command_body: Tensor,
    schedule,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    trajectory = torch.as_tensor(state)
    foot = go2_fk(trajectory[..., :3], trajectory[..., 3:6], trajectory[..., 6:]).foot_pos_w
    foot_step_xy = foot[:, 1:, :, :2] - foot[:, :-1, :, :2]
    root_step_xy = trajectory[:, 1:, :2] - trajectory[:, :-1, :2]
    activity_scale = float(cfg.loss_terms.command_activity_scale)
    foot_step = directional_progress(
        foot_step_xy,
        command_body,
        trajectory[:, :-1, 5],
        activity_scale=activity_scale,
    )
    root_step = directional_progress(
        root_step_xy,
        command_body,
        trajectory[:, :-1, 5],
        activity_scale=activity_scale,
    )
    command_norm = torch.linalg.vector_norm(torch.as_tensor(command_body, dtype=trajectory.dtype, device=trajectory.device)[:, :2], dim=-1)
    activity = 1.0 - torch.exp(-command_norm.square() / activity_scale**2)
    swing = torch.logical_and(schedule.swing[:, 1:], schedule.swing[:, :-1]).to(trajectory.dtype)
    speed_scale = float(cfg.loss_terms.swing_speed_command_scale)
    margin_scale = (command_norm / speed_scale).clamp(0.0, 1.0)
    margin = (
        float(cfg.loss_terms.swing_speed_margin)
        * margin_scale[:, None, None]
        * activity[:, None, None]
    )
    penalty = torch.relu(root_step[..., None] + margin - foot_step) * swing
    early_phase = (1.0 - schedule.swing_tau[:, :-1]).to(trajectory.dtype) * swing
    early_weight = float(cfg.loss_terms.swing_speed_early)
    phase_weight = 1.0 + (early_weight - 1.0) * early_phase
    penalty = penalty * phase_weight.clamp_min(0.0).sqrt()
    denominator = swing.sum(dim=(1, 2)).clamp_min(1.0).sqrt()
    return penalty.flatten(1) / denominator[:, None]


def swing_speed_loss(state: Tensor, command_body: Tensor, schedule, cfg: JointMpcRtiCfg) -> Tensor:
    residual = swing_speed_residual(state, command_body, schedule, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["directional_progress", "swing_speed_loss", "swing_speed_penalty", "swing_speed_residual"]
