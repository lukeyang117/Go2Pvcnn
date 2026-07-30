from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.parallelism.config import OfficialCollisionShapeSpec, ParallelismCfg
from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismTerrain


LEG_NAMES = ("FL", "FR", "RL", "RR")


def _tensor(reference: Tensor, values) -> Tensor:
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def _quat_to_matrix(quat_wxyz: Tensor) -> Tensor:
    quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = quat.unbind(dim=-1)
    two = torch.tensor(2.0, dtype=quat.dtype, device=quat.device)
    row0 = torch.stack((1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), dim=-1)
    row1 = torch.stack((two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)), dim=-1)
    row2 = torch.stack((two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _box_surface_points(spec: OfficialCollisionShapeSpec, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    half = torch.tensor(spec.size_l, dtype=dtype, device=device) * 0.5
    zeros = torch.zeros((), dtype=dtype, device=device)
    return torch.stack(
        (
            torch.stack((half[0], zeros, zeros)),
            torch.stack((-half[0], zeros, zeros)),
            torch.stack((zeros, half[1], zeros)),
            torch.stack((zeros, -half[1], zeros)),
            torch.stack((zeros, zeros, half[2])),
            torch.stack((zeros, zeros, -half[2])),
        ),
        dim=0,
    )


def _cylinder_surface_points(spec: OfficialCollisionShapeSpec, cfg: ParallelismCfg, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    half_h = torch.tensor(float(spec.height_m) * 0.5, dtype=dtype, device=device)
    radius = torch.tensor(float(spec.radius_m), dtype=dtype, device=device)
    zeros = torch.zeros((), dtype=dtype, device=device)
    points = [
        torch.stack((-half_h, zeros, zeros)),
        torch.stack((half_h, zeros, zeros)),
    ]
    layers = max(1, int(cfg.cylinder_layers))
    if layers == 1:
        x_values = torch.zeros(1, dtype=dtype, device=device)
    else:
        x_values = torch.linspace(-float(spec.height_m) * 0.5, float(spec.height_m) * 0.5, layers, dtype=dtype, device=device)
    angles = max(4, int(cfg.cylinder_angles))
    for x_value in x_values:
        for angle_idx in range(angles):
            angle = torch.tensor((2.0 * math.pi * angle_idx) / angles, dtype=dtype, device=device)
            points.append(torch.stack((x_value, radius * torch.cos(angle), radius * torch.sin(angle))))
    return torch.stack(points, dim=0)


def _sphere_surface_points(spec: OfficialCollisionShapeSpec, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    radius = torch.tensor(float(spec.radius_m), dtype=dtype, device=device)
    zeros = torch.zeros((), dtype=dtype, device=device)
    return torch.stack(
        (
            torch.stack((radius, zeros, zeros)),
            torch.stack((-radius, zeros, zeros)),
            torch.stack((zeros, radius, zeros)),
            torch.stack((zeros, -radius, zeros)),
            torch.stack((zeros, zeros, radius)),
            torch.stack((zeros, zeros, -radius)),
        ),
        dim=0,
    )


def _primitive_surface_points(spec: OfficialCollisionShapeSpec, cfg: ParallelismCfg, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if spec.shape_type == "box":
        return _box_surface_points(spec, dtype=dtype, device=device)
    if spec.shape_type == "cylinder":
        # Initial design uses two end centers plus one four-direction ring.
        return _cylinder_surface_points(spec, cfg, dtype=dtype, device=device)
    if spec.shape_type == "sphere":
        _ = cfg.sphere_surface_points
        return _sphere_surface_points(spec, dtype=dtype, device=device)
    raise ValueError(f"unsupported official collision shape_type: {spec.shape_type}")


def build_official_surface_points_l(
    specs: tuple[OfficialCollisionShapeSpec, ...],
    cfg: ParallelismCfg,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    point_sets = []
    max_points = 0
    for spec in specs:
        primitive_points = _primitive_surface_points(spec, cfg, dtype=dtype, device=device)
        quat = torch.tensor(spec.quat_wxyz_l, dtype=dtype, device=device)
        rot = _quat_to_matrix(quat)
        center = torch.tensor(spec.center_l, dtype=dtype, device=device)
        link_points = torch.matmul(rot, primitive_points[..., None]).squeeze(-1) + center
        point_sets.append(link_points)
        max_points = max(max_points, int(link_points.shape[0]))

    padded = torch.zeros(len(specs), max_points, 3, dtype=dtype, device=device)
    mask = torch.zeros(len(specs), max_points, dtype=torch.bool, device=device)
    for shape_idx, points in enumerate(point_sets):
        count = int(points.shape[0])
        padded[shape_idx, :count] = points
        mask[shape_idx, :count] = True
    return padded, mask


def _same_leg_pose(pos_w: Tensor, rot_w: Tensor) -> tuple[Tensor, Tensor]:
    batch, leg_count, candidate_count = pos_w.shape[:3]
    point_index = torch.arange(leg_count, device=pos_w.device).view(1, leg_count, 1, 1, 1)
    pos_index = point_index.expand(batch, leg_count, candidate_count, 1, 3)
    rot_index = point_index[..., None].expand(batch, leg_count, candidate_count, 1, 3, 3)
    return pos_w.gather(3, pos_index).squeeze(3), rot_w.gather(3, rot_index).squeeze(3)


def _link_pose_for_group(geometry, link_type: str) -> tuple[Tensor, Tensor]:
    if link_type == "thigh":
        return _same_leg_pose(geometry.thigh_pos_w, geometry.thigh_rot_w)
    if link_type == "calf":
        return _same_leg_pose(geometry.calf_pos_w, geometry.calf_rot_w)
    if link_type == "foot":
        return _same_leg_pose(geometry.foot_pos_w, geometry.foot_rot_w)
    raise ValueError(f"unsupported official collision link_type: {link_type}")


def _group_indices(specs: tuple[OfficialCollisionShapeSpec, ...], link_type: str) -> list[int]:
    return [idx for idx, spec in enumerate(specs) if spec.link_type == link_type]


def _leg_specific_mask(specs: tuple[OfficialCollisionShapeSpec, ...], leg_count: int, *, device: torch.device) -> Tensor:
    allowed = torch.ones(leg_count, len(specs), dtype=torch.bool, device=device)
    for shape_idx, spec in enumerate(specs):
        if spec.leg_name is None:
            continue
        if spec.leg_name.startswith("!"):
            excluded = spec.leg_name[1:]
            for leg_idx, leg_name in enumerate(LEG_NAMES[:leg_count]):
                allowed[leg_idx, shape_idx] = leg_name != excluded
            continue
        for leg_idx, leg_name in enumerate(LEG_NAMES[:leg_count]):
            allowed[leg_idx, shape_idx] = leg_name == spec.leg_name
    return allowed


def _official_group_collision(
    terrain: ParallelismTerrain,
    geometry,
    cfg: ParallelismCfg,
    specs: tuple[OfficialCollisionShapeSpec, ...],
    indices: list[int],
) -> Tensor:
    group_specs = tuple(specs[idx] for idx in indices)
    link_type = group_specs[0].link_type
    link_pos_w, link_rot_w = _link_pose_for_group(geometry, link_type)
    batch, leg_count, candidate_count = link_pos_w.shape[:3]
    points_l, point_mask = build_official_surface_points_l(group_specs, cfg, dtype=link_pos_w.dtype, device=link_pos_w.device)
    group_count, point_count = points_l.shape[:2]

    link_pos_g = link_pos_w[:, :, :, None, None, :]
    link_rot_g = link_rot_w[:, :, :, None, None, :, :]
    points_w = torch.matmul(link_rot_g, points_l.view(1, 1, 1, group_count, point_count, 3, 1)).squeeze(-1) + link_pos_g
    query = query_height_semantic_valid(
        terrain,
        points_w[..., :2].reshape(batch, leg_count * candidate_count * group_count * point_count, 2),
    )
    terrain_h = query.height.reshape(batch, leg_count, candidate_count, group_count, point_count)
    terrain_valid = query.valid.reshape(batch, leg_count, candidate_count, group_count, point_count)
    semantic = query.semantic.reshape(batch, leg_count, candidate_count, group_count, point_count)
    obstacle_ids = torch.tensor(tuple(cfg.obstacle_semantic_ids), dtype=semantic.dtype, device=semantic.device)
    semantic_hit = (semantic[..., None] == obstacle_ids).any(dim=-1)
    terrain_hit = terrain_h >= (points_w[..., 2] - float(cfg.collision_margin_m))
    tolerant_names = set(cfg.contact_tolerant_collision_shape_names)
    terrain_checked = torch.tensor(
        tuple(spec.name not in tolerant_names for spec in group_specs),
        dtype=torch.bool,
        device=points_w.device,
    ).view(1, 1, 1, group_count, 1)
    terrain_hit = terrain_hit & terrain_checked
    valid_point = point_mask.view(1, 1, 1, group_count, point_count)
    point_hit = valid_point & ((~terrain_valid) | semantic_hit | terrain_hit)
    return point_hit.any(dim=-1)


def official_collision_mask(
    terrain: ParallelismTerrain,
    geometry,
    cfg: ParallelismCfg,
) -> tuple[Tensor, Tensor]:
    specs = tuple(cfg.official_collision_shapes)
    batch, leg_count, candidate_count = geometry.foot_pos_w.shape[:3]
    collision_bits = torch.zeros(
        batch,
        leg_count,
        candidate_count,
        len(specs),
        dtype=torch.bool,
        device=geometry.foot_pos_w.device,
    )
    candidate_hit = torch.zeros(batch, leg_count, candidate_count, dtype=torch.bool, device=geometry.foot_pos_w.device)
    allowed_by_leg = _leg_specific_mask(specs, leg_count, device=geometry.foot_pos_w.device)
    for link_type in ("thigh", "calf", "foot"):
        indices = _group_indices(specs, link_type)
        if not indices:
            continue
        group_hit = _official_group_collision(terrain, geometry, cfg, specs, indices)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=collision_bits.device)
        allowed = allowed_by_leg[:, indices].view(1, leg_count, 1, len(indices))
        group_hit = group_hit & allowed
        collision_bits.index_copy_(dim=-1, index=index_tensor, source=group_hit)
        candidate_hit = torch.logical_or(candidate_hit, torch.any(group_hit, dim=-1))
    return ~candidate_hit, collision_bits
