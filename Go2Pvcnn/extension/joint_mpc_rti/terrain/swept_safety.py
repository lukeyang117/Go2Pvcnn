"""Whole-robot node and swept-interval safety on the current perceptive field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.go2_kinematics import go2_collision_geometry
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import (
    query_inflated_height_world,
    query_perceptive_world,
)
from extension.joint_mpc_rti.types import JointMpcPerceptiveField


PART_NAMES = ("foot", "knee", "calf", "thigh", "base")


@dataclass(frozen=True)
class WholeRobotSafety:
    safe: Tensor
    minimum_clearance: Tensor
    minimum_clearance_by_part: dict[str, Tensor]
    collision_by_part: dict[str, Tensor]
    valid_by_part: dict[str, Tensor]
    sole_safe: Tensor
    underresolved: Tensor


def _sample_capsule(endpoints_w: Tensor, samples: int) -> Tensor:
    reference = endpoints_w
    fractions = constant_like(
        reference,
        f"capsule_fractions_{samples}",
        tuple(index / float(samples - 1) for index in range(samples)),
    )
    shape = *((1,) * (endpoints_w.ndim - 2)), samples, 1
    fraction = fractions.view(shape)
    start = endpoints_w[..., 0, :].unsqueeze(-2)
    end = endpoints_w[..., 1, :].unsqueeze(-2)
    return start + fraction * (end - start)


def _part_clearance(
    points_w: Tensor,
    field: JointMpcPerceptiveField,
    *,
    channel: int,
    vertical_margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, nodes = int(points_w.shape[0]), int(points_w.shape[1])
    points_per_node = int(points_w.numel() // (batch * nodes * 3))
    points = points_w.reshape(batch, nodes * points_per_node, 3)
    inflated_height, valid_query = query_inflated_height_world(
        field, points, channel=channel
    )
    clearance = (
        points[..., 2]
        - inflated_height
        - float(vertical_margin)
    ).reshape(batch, nodes, points_per_node)
    valid = valid_query.reshape(batch, nodes, points_per_node)
    finite_clearance = torch.where(
        valid,
        clearance,
        torch.full_like(clearance, -torch.inf),
    )
    minimum = finite_clearance.amin(dim=-1)
    all_valid = valid.all(dim=-1)
    collision = (~all_valid) | (minimum < 0.0)
    return minimum, all_valid, collision


def _base_clearance(
    points_w: Tensor,
    field: JointMpcPerceptiveField,
    *,
    margin: float,
    wall_height: float,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, nodes, samples = map(int, points_w.shape[:3])
    points = points_w.reshape(batch, nodes * samples, 3)
    query = query_perceptive_world(field, points)
    effective_height = torch.where(
        query.large_mask | query.unknown_mask,
        query.height_w.new_full((), float(wall_height)),
        query.height_w,
    )
    clearance = (points[..., 2] - effective_height - float(margin)).reshape(
        batch, nodes, samples
    )
    valid = query.valid.reshape(batch, nodes, samples)
    finite_clearance = torch.where(
        valid,
        clearance,
        torch.full_like(clearance, -torch.inf),
    )
    minimum = finite_clearance.amin(dim=-1)
    all_valid = valid.all(dim=-1)
    collision = (~all_valid) | (minimum < 0.0)
    return minimum, all_valid, collision


def _sole_support_safe(
    sole_corners_w: Tensor,
    field: JointMpcPerceptiveField,
    contact_state: Tensor | None,
    *,
    ground_tolerance: float,
) -> tuple[Tensor, Tensor]:
    batch, nodes = map(int, sole_corners_w.shape[:2])
    center_w = sole_corners_w.mean(dim=-2, keepdim=True)
    sole_samples = torch.cat((sole_corners_w, center_w), dim=-2)
    points = sole_samples.reshape(batch, nodes * 20, 3)
    query = query_perceptive_world(field, points)
    sample_shape = (batch, nodes, 4, 5)
    valid_samples = query.valid.reshape(sample_shape)
    raw_safe = (
        valid_samples[..., :4]
        & ~query.small_mask.reshape(sample_shape)[..., :4]
        & ~query.large_mask.reshape(sample_shape)[..., :4]
        & ~query.unknown_mask.reshape(sample_shape)[..., :4]
        & ~query.semantic_edge_mask.reshape(sample_shape)[..., :4]
    ).all(dim=-1)
    center_safe = query.landing_safe.reshape(sample_shape)[..., 4]
    landing = raw_safe & center_safe
    valid = valid_samples.all(dim=-1)
    ground_error = (
        points[..., 2] - query.height_w
    ).reshape(sample_shape)
    on_ground = (
        ground_error[..., :4].abs() <= float(ground_tolerance)
    ).all(dim=-1)
    sole_safe = landing & valid & on_ground
    if contact_state is None:
        return torch.ones_like(sole_safe), torch.zeros(batch, nodes, dtype=torch.bool, device=points.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=points.device)
    if contact.shape != (batch, nodes, 4):
        raise ValueError("contact_state must have shape [B,N,4]")
    support_collision = (contact & ~sole_safe).any(dim=-1)
    return torch.where(contact, sole_safe, torch.ones_like(sole_safe)), support_collision


def evaluate_nodes(
    trajectory_nodes: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    contact_state: Tensor | None = None,
) -> WholeRobotSafety:
    state = torch.as_tensor(trajectory_nodes)
    if state.ndim != 3 or int(state.shape[-1]) != 18:
        raise ValueError("trajectory_nodes must have shape [B,N,18]")
    geometry = go2_collision_geometry(state[..., :3], state[..., 3:6], state[..., 6:])
    terrain = cfg.terrain
    capsule_samples = int(terrain.capsule_samples)

    foot = _part_clearance(
        geometry.foot_center_w,
        field,
        channel=0,
        vertical_margin=float(terrain.foot_radius_m),
    )
    knee = _part_clearance(
        geometry.knee_center_w,
        field,
        channel=1,
        vertical_margin=float(terrain.knee_radius_m + terrain.link_margin_m),
    )
    calf = _part_clearance(
        _sample_capsule(geometry.calf_endpoints_w, capsule_samples),
        field,
        channel=2,
        vertical_margin=float(terrain.calf_radius_m + terrain.link_margin_m),
    )
    thigh = _part_clearance(
        _sample_capsule(geometry.thigh_endpoints_w, capsule_samples),
        field,
        channel=3,
        vertical_margin=float(terrain.thigh_radius_m + terrain.link_margin_m),
    )
    base = _base_clearance(
        geometry.base_bottom_samples_w,
        field,
        margin=float(terrain.base_margin_m),
        wall_height=float(terrain.h_wall),
    )
    values = dict(zip(PART_NAMES, (foot, knee, calf, thigh, base), strict=True))
    minimum_by_part = {name: value[0] for name, value in values.items()}
    valid_by_part = {name: value[1] for name, value in values.items()}
    collision_by_part = {name: value[2] for name, value in values.items()}
    sole_safe, support_collision = _sole_support_safe(
        geometry.sole_corners_w,
        field,
        contact_state,
        ground_tolerance=float(terrain.stance_ground_tolerance_m),
    )
    collision_by_part["foot"] = collision_by_part["foot"] | support_collision
    stacked_clearance = torch.stack(tuple(minimum_by_part.values()), dim=-1)
    collision = torch.stack(tuple(collision_by_part.values()), dim=-1).any(dim=-1)
    finite = torch.isfinite(state).all(dim=-1)
    return WholeRobotSafety(
        safe=finite & ~collision,
        minimum_clearance=stacked_clearance.amin(dim=-1),
        minimum_clearance_by_part=minimum_by_part,
        collision_by_part=collision_by_part,
        valid_by_part=valid_by_part,
        sole_safe=sole_safe,
        underresolved=torch.zeros_like(collision),
    )


def evaluate_swept_intervals(
    trajectory_nodes: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    contact_state: Tensor | None = None,
) -> WholeRobotSafety:
    state = torch.as_tensor(trajectory_nodes)
    if state.ndim != 3 or int(state.shape[-1]) != 18 or int(state.shape[1]) < 2:
        raise ValueError("trajectory_nodes must have shape [B,N>=2,18]")
    subdivisions = int(cfg.terrain.sweep_subdivisions)
    samples = subdivisions + 1
    fractions = constant_like(
        state,
        f"sweep_fractions_{subdivisions}",
        tuple(index / float(subdivisions) for index in range(samples)),
    ).view(1, 1, samples, 1)
    swept = state[:, :-1, None] + fractions * (state[:, 1:, None] - state[:, :-1, None])
    batch, intervals = int(swept.shape[0]), int(swept.shape[1])
    swept_contact = None
    if contact_state is not None:
        contact = torch.as_tensor(contact_state, dtype=torch.bool, device=state.device)
        if contact.shape != state.shape[:2] + (4,):
            raise ValueError("contact_state must have shape [B,N,4]")
        swept_contact = contact[:, :-1, None].expand(-1, -1, samples, -1).reshape(
            batch, intervals * samples, 4
        )
    sampled = evaluate_nodes(
        swept.reshape(batch, intervals * samples, 18),
        field,
        cfg,
        contact_state=swept_contact,
    )

    minimum_by_part = {
        name: value.reshape(batch, intervals, samples).amin(dim=-1)
        for name, value in sampled.minimum_clearance_by_part.items()
    }
    collision_by_part = {
        name: value.reshape(batch, intervals, samples).any(dim=-1)
        for name, value in sampled.collision_by_part.items()
    }
    valid_by_part = {
        name: value.reshape(batch, intervals, samples).all(dim=-1)
        for name, value in sampled.valid_by_part.items()
    }
    stacked_clearance = torch.stack(tuple(minimum_by_part.values()), dim=-1)
    collision = torch.stack(tuple(collision_by_part.values()), dim=-1).any(dim=-1)
    delta = state[:, 1:] - state[:, :-1]
    translation_bound = torch.linalg.vector_norm(delta[..., :3], dim=-1)
    rotation_bound = 0.35 * torch.linalg.vector_norm(delta[..., 3:6], dim=-1)
    joint_delta = delta[..., 6:].reshape(batch, intervals, 4, 3).abs().sum(dim=-1).amax(dim=-1)
    joint_bound = 0.426 * joint_delta
    sample_motion_bound = (
        translation_bound + rotation_bound + joint_bound
    ) / float(subdivisions)
    underresolved = sample_motion_bound > float(field.resolution)
    return WholeRobotSafety(
        safe=~collision & ~underresolved,
        minimum_clearance=stacked_clearance.amin(dim=-1),
        minimum_clearance_by_part=minimum_by_part,
        collision_by_part=collision_by_part,
        valid_by_part=valid_by_part,
        sole_safe=sampled.sole_safe.reshape(batch, intervals, samples, 4).all(dim=2),
        underresolved=underresolved,
    )


__all__ = [
    "PART_NAMES",
    "WholeRobotSafety",
    "evaluate_nodes",
    "evaluate_swept_intervals",
]
