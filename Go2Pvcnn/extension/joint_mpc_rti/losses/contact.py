"""World-frame stance locking and physical ground-contact residuals."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import query_world


def contact_residual(state: Tensor, context, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    foot = go2_fk(trajectory[..., :3], trajectory[..., 3:6], trajectory[..., 6:]).foot_pos_w
    stance = context.schedule.stance.to(trajectory.dtype)
    anchor = torch.as_tensor(context.stance_anchor_w, dtype=trajectory.dtype, device=trajectory.device)
    query = query_world(context.terrain, foot.reshape(trajectory.shape[0], -1, 3))
    height = query.height_w.reshape_as(foot[..., 2])
    valid = query.valid.reshape_as(foot[..., 2]).to(trajectory.dtype)
    current_segment = torch.cumprod(context.schedule.stance.to(torch.int64), dim=1).to(torch.bool)
    previous_foot = torch.cat((foot[:, :1], foot[:, :-1]), dim=1)
    previous_stance = torch.cat(
        (torch.zeros_like(context.schedule.stance[:, :1]), context.schedule.stance[:, :-1]),
        dim=1,
    )
    future_onset = context.schedule.stance & ~previous_stance & ~current_segment
    future_continuing = context.schedule.stance & previous_stance & ~current_segment
    anchor_error = torch.where(
        current_segment[..., None],
        foot[..., :2] - anchor[..., :2],
        foot[..., :2] - previous_foot[..., :2],
    )
    touchdown = torch.as_tensor(
        context.touchdown_reference_w, dtype=trajectory.dtype, device=trajectory.device
    )
    xy_error = torch.where(
        future_onset[..., None],
        foot[..., :2] - touchdown[..., :2],
        anchor_error,
    )
    strong_scale = math.sqrt(float(cfg.loss_terms.contact_anchor_xy))
    onset_scale = math.sqrt(float(cfg.loss_terms.contact_future_onset_xy))
    xy_scales = constant_like(
        trajectory,
        f"contact_xy_scales_{onset_scale}_{strong_scale}",
        (onset_scale, strong_scale),
    )
    xy_scale = torch.where(
        future_onset,
        xy_scales[0],
        xy_scales[1],
    )
    future_active = future_onset | future_continuing
    xy_active = current_segment | future_active
    xy = xy_scale[..., None] * xy_error
    xy = xy * xy_active[..., None]
    ground = math.sqrt(float(cfg.loss_terms.contact_ground)) * (
        foot[..., 2] - height - float(cfg.gait.foot_contact_offset)
    )
    ground = ground * stance * valid
    current_xy_denominator = (current_segment.sum(dim=(1, 2)) * 2.0).clamp_min(1.0).sqrt()
    future_xy_denominator = (xy_active.sum(dim=(1, 2)) * 2.0).clamp_min(1.0).sqrt()
    xy_denominator = torch.where(
        current_segment,
        current_xy_denominator[:, None, None],
        future_xy_denominator[:, None, None],
    )
    z_denominator = stance.sum(dim=(1, 2)).clamp_min(1.0).sqrt()
    return torch.cat(
        (
            (xy / xy_denominator[..., None]).flatten(1),
            ground.flatten(1) / z_denominator[:, None],
        ),
        dim=1,
    )


def contact_loss(state: Tensor, context, cfg: JointMpcRtiCfg) -> Tensor:
    residual = contact_residual(state, context, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["contact_loss", "contact_residual"]
