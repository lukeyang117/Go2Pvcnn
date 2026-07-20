"""Unified differentiable full-body terrain and semantic clearance residual."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.terrain.query import query_world


def _surface_height(
    raw_height: Tensor,
    small_occupancy: Tensor,
    large_occupancy: Tensor,
    small_height: Tensor,
    *,
    small_wall: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    wall = torch.full_like(raw_height, float(cfg.terrain.h_wall))
    small_target = torch.where(small_wall, wall, small_height)
    small_surface = torch.lerp(raw_height, small_target, small_occupancy)
    return torch.lerp(small_surface, wall, large_occupancy)


def _clearance_residual(position: Tensor, surface: Tensor, margin: float, temperature: float) -> Tensor:
    deficit = surface + float(margin) - position[..., 2]
    return float(temperature) * F.softplus(deficit / float(temperature))


def terrain_residual(state: Tensor, context, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    batch, nodes = trajectory.shape[:2]
    geometry = go2_fk(trajectory[..., :3], trajectory[..., 3:6], trajectory[..., 6:])
    calf = geometry.shank_samples_w.reshape(batch, nodes, 12, 3)
    thigh = geometry.thigh_samples_w.reshape(batch, nodes, 12, 3)
    points = torch.cat(
        (
            geometry.foot_pos_w,
            geometry.knee_pos_w,
            calf,
            thigh,
            geometry.body_samples_w,
        ),
        dim=2,
    )
    query = query_world(context.terrain, points.reshape(batch, nodes * 41, 3))
    query_shape = (batch, nodes, 41)
    height = query.height_w.reshape(query_shape)
    small_occupancy = query.small_occupancy.reshape(query_shape)
    large_occupancy = query.large_occupancy.reshape(query_shape)
    small_height = query.small_propagated_height.reshape(query_shape)
    foot_stance = context.schedule.stance
    never_wall = torch.zeros(batch, nodes, 1, dtype=torch.bool, device=trajectory.device)
    foot_surface = _surface_height(
        height[..., :4],
        small_occupancy[..., :4],
        large_occupancy[..., :4],
        small_height[..., :4],
        small_wall=foot_stance,
        cfg=cfg,
    )
    all_surface = _surface_height(
        height,
        small_occupancy,
        large_occupancy,
        small_height,
        small_wall=never_wall,
        cfg=cfg,
    )
    temperature = float(cfg.loss_terms.terrain_temperature)
    foot = _clearance_residual(
        geometry.foot_pos_w,
        foot_surface,
        0.022 + float(cfg.loss_terms.terrain_foot_margin),
        temperature,
    )
    knee = _clearance_residual(
        geometry.knee_pos_w,
        all_surface[..., 4:8],
        0.040 + float(cfg.loss_terms.terrain_link_margin),
        temperature,
    )
    calf_residual = _clearance_residual(
        calf,
        all_surface[..., 8:20],
        0.040 + float(cfg.loss_terms.terrain_link_margin),
        temperature,
    )
    thigh_residual = _clearance_residual(
        thigh,
        all_surface[..., 20:32],
        0.040 + float(cfg.loss_terms.terrain_link_margin),
        temperature,
    )
    base = _clearance_residual(
        geometry.body_samples_w,
        all_surface[..., 32:41],
        float(cfg.loss_terms.terrain_base_margin),
        temperature,
    )
    touchdown = math.sqrt(float(cfg.loss_terms.terrain_touchdown_avoidance)) * (
        small_occupancy[..., :4] * foot_stance.to(trajectory.dtype)
    )
    residual = torch.cat(
        (
            foot.flatten(1),
            knee.flatten(1),
            calf_residual.flatten(1),
            thigh_residual.flatten(1),
            base.flatten(1),
            touchdown.flatten(1),
        ),
        dim=1,
    )
    return residual / math.sqrt(float(residual.shape[1]))


def terrain_loss(state: Tensor, context, cfg: JointMpcRtiCfg) -> Tensor:
    residual = terrain_residual(state, context, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["terrain_loss", "terrain_residual"]
