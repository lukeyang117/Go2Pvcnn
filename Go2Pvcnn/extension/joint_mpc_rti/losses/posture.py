"""Root support pose and nominal joint posture residuals."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg


def posture_residual(state: Tensor, support_height: Tensor, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    support = torch.as_tensor(support_height, dtype=trajectory.dtype, device=trajectory.device)
    nominal_joint = trajectory.new_tensor(cfg.gait.nominal_joint_pos).view(1, 1, 12)
    height = math.sqrt(float(cfg.loss_terms.posture_root_height)) * (
        trajectory[..., 2] - support - float(cfg.loss_terms.posture_root_clearance)
    )
    rpy = math.sqrt(float(cfg.loss_terms.posture_roll_pitch)) * trajectory[..., 3:5]
    joint = math.sqrt(float(cfg.loss_terms.posture_joint)) * (trajectory[..., 6:] - nominal_joint)
    residual = torch.cat((height[..., None], rpy, joint), dim=-1)
    return residual.flatten(1) / math.sqrt(float(residual[0].numel()))


def posture_loss(state: Tensor, support_height: Tensor, cfg: JointMpcRtiCfg) -> Tensor:
    residual = posture_residual(state, support_height, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["posture_loss", "posture_residual"]
