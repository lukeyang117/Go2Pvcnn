"""World-frame stance locking and physical ground-contact residuals."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.terrain.query import query_world


def contact_residual(state: Tensor, context, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    foot = go2_fk(trajectory[..., :3], trajectory[..., 3:6], trajectory[..., 6:]).foot_pos_w
    stance = context.schedule.stance.to(trajectory.dtype)
    anchor = torch.as_tensor(context.stance_anchor_w, dtype=trajectory.dtype, device=trajectory.device)
    query = query_world(context.terrain, foot.reshape(trajectory.shape[0], -1, 3))
    height = query.height_w.reshape_as(foot[..., 2])
    valid = query.valid.reshape_as(foot[..., 2]).to(trajectory.dtype)
    xy = math.sqrt(float(cfg.loss_terms.contact_anchor_xy)) * (foot[..., :2] - anchor[..., :2])
    xy = xy * stance[..., None]
    ground = math.sqrt(float(cfg.loss_terms.contact_ground)) * (
        foot[..., 2] - height - float(cfg.gait.foot_contact_offset)
    )
    ground = ground * stance * valid
    xy_denominator = (stance.sum(dim=(1, 2)) * 2.0).clamp_min(1.0).sqrt()
    z_denominator = stance.sum(dim=(1, 2)).clamp_min(1.0).sqrt()
    return torch.cat(
        (xy.flatten(1) / xy_denominator[:, None], ground.flatten(1) / z_denominator[:, None]),
        dim=1,
    )


def contact_loss(state: Tensor, context, cfg: JointMpcRtiCfg) -> Tensor:
    residual = contact_residual(state, context, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["contact_loss", "contact_residual"]
