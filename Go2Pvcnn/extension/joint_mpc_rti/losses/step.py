"""Command-conditioned touchdown reference residuals."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk


def step_residual(state: Tensor, touchdown_reference_w: Tensor, schedule, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    target = torch.as_tensor(touchdown_reference_w, dtype=trajectory.dtype, device=trajectory.device)
    foot = go2_fk(trajectory[..., :3], trajectory[..., 3:6], trajectory[..., 6:]).foot_pos_w
    event = schedule.phase.eq(int(cfg.gait.swing_steps)).to(trajectory.dtype)
    scale = torch.stack(
        (
            trajectory.new_tensor(math.sqrt(float(cfg.loss_terms.step_xy))),
            trajectory.new_tensor(math.sqrt(float(cfg.loss_terms.step_xy))),
            trajectory.new_tensor(math.sqrt(float(cfg.loss_terms.step_z))),
        )
    )
    residual = (foot - target) * event[..., None] * scale
    denominator = (event.sum(dim=(1, 2)) * 3.0).clamp_min(1.0).sqrt()
    return residual.flatten(1) / denominator[:, None]


def step_loss(state: Tensor, touchdown_reference_w: Tensor, schedule, cfg: JointMpcRtiCfg) -> Tensor:
    residual = step_residual(state, touchdown_reference_w, schedule, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["step_loss", "step_residual"]
