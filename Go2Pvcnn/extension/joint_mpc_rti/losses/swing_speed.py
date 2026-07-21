"""Swing-foot progress relative to root progress."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk


def swing_speed_penalty(foot_step: Tensor, root_step: Tensor, *, margin: float) -> Tensor:
    return torch.relu(torch.as_tensor(root_step) + float(margin) - torch.as_tensor(foot_step))


def swing_speed_residual(state: Tensor, schedule, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    foot = go2_fk(trajectory[..., :3], trajectory[..., 3:6], trajectory[..., 6:]).foot_pos_w
    foot_step = torch.sqrt((foot[:, 1:, :, :2] - foot[:, :-1, :, :2]).square().sum(dim=-1) + 1.0e-12)
    root_step = torch.sqrt((trajectory[:, 1:, :2] - trajectory[:, :-1, :2]).square().sum(dim=-1) + 1.0e-12)
    swing = torch.logical_and(schedule.swing[:, 1:], schedule.swing[:, :-1]).to(trajectory.dtype)
    penalty = swing_speed_penalty(
        foot_step,
        root_step[..., None],
        margin=float(cfg.loss_terms.swing_speed_margin),
    ) * swing
    early_phase = (1.0 - schedule.swing_tau[:, :-1]).to(trajectory.dtype) * swing
    early_weight = float(cfg.loss_terms.swing_speed_early)
    phase_weight = 1.0 + (early_weight - 1.0) * early_phase
    penalty = penalty * phase_weight.clamp_min(0.0).sqrt()
    denominator = swing.sum(dim=(1, 2)).clamp_min(1.0).sqrt()
    return penalty.flatten(1) / denominator[:, None]


def swing_speed_loss(state: Tensor, schedule, cfg: JointMpcRtiCfg) -> Tensor:
    residual = swing_speed_residual(state, schedule, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["swing_speed_loss", "swing_speed_penalty", "swing_speed_residual"]
