from __future__ import annotations

import torch
from torch import Tensor

from extension.parallelism.config import EllipsoidSpec, ParallelismCfg
from extension.parallelism.kinematics import LEG_SIDE_SIGNS
from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismTerrain


def _tensor(reference: Tensor, values) -> Tensor:
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def build_ellipsoid_probe_l(
    specs: tuple[EllipsoidSpec, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    centers = torch.tensor(tuple(spec.center_l for spec in specs), dtype=dtype, device=device)
    offsets = torch.tensor(tuple(spec.probe_offset_l for spec in specs), dtype=dtype, device=device)
    zeros = torch.zeros(offsets.shape[0], dtype=dtype, device=device)
    probe_offsets = torch.stack(
        (
            torch.stack((zeros, zeros, zeros), dim=-1),
            torch.stack((offsets[:, 0], zeros, zeros), dim=-1),
            torch.stack((-offsets[:, 0], zeros, zeros), dim=-1),
            torch.stack((zeros, offsets[:, 1], zeros), dim=-1),
            torch.stack((zeros, -offsets[:, 1], zeros), dim=-1),
        ),
        dim=1,
    )
    return centers[:, None, :] + probe_offsets


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
    raise ValueError(f"unsupported ellipsoid link_type: {link_type}")


def _group_indices(specs: tuple[EllipsoidSpec, ...], link_type: str) -> list[int]:
    return [idx for idx, spec in enumerate(specs) if spec.link_type == link_type]


def _mirror_y_by_leg(values_l: Tensor, leg_count: int) -> Tensor:
    side_shape = (1, leg_count, 1) + ((1,) * values_l.ndim)
    side = _tensor(values_l, LEG_SIDE_SIGNS[:leg_count]).view(*side_shape)
    expand_shape = (-1, leg_count, -1) + ((-1,) * values_l.ndim)
    mirrored = values_l.view(1, 1, 1, *values_l.shape).expand(*expand_shape).clone()
    mirrored[..., 1:2] = mirrored[..., 1:2] * side
    return mirrored


def _ellipsoid_group_collision(
    terrain: ParallelismTerrain,
    geometry,
    cfg: ParallelismCfg,
    specs: tuple[EllipsoidSpec, ...],
    indices: list[int],
) -> Tensor:
    group_specs = tuple(specs[idx] for idx in indices)
    link_type = group_specs[0].link_type
    link_pos_w, link_rot_w = _link_pose_for_group(geometry, link_type)
    batch, leg_count, candidate_count = link_pos_w.shape[:3]
    probe_l_raw = build_ellipsoid_probe_l(group_specs, dtype=link_pos_w.dtype, device=link_pos_w.device)
    center_l_raw = torch.tensor(tuple(spec.center_l for spec in group_specs), dtype=link_pos_w.dtype, device=link_pos_w.device)
    radii_l = torch.tensor(tuple(spec.radii_l for spec in group_specs), dtype=link_pos_w.dtype, device=link_pos_w.device)

    probe_l = _mirror_y_by_leg(probe_l_raw, leg_count)
    center_l = _mirror_y_by_leg(center_l_raw, leg_count)
    group_count = len(group_specs)

    link_pos_g = link_pos_w[:, :, :, None, None, :]
    link_rot_g = link_rot_w[:, :, :, None, None, :, :]
    probe_w = torch.matmul(link_rot_g, probe_l[..., None]).squeeze(-1) + link_pos_g
    query = query_height_semantic_valid(
        terrain,
        probe_w[..., :2].reshape(batch, leg_count * candidate_count * group_count * int(cfg.collision_probe_count), 2),
    )
    terrain_h = query.height.reshape(batch, leg_count, candidate_count, group_count, int(cfg.collision_probe_count))
    terrain_p_w = torch.stack((probe_w[..., 0], probe_w[..., 1], terrain_h), dim=-1)
    terrain_delta_w = terrain_p_w - link_pos_w[:, :, :, None, None, :]
    terrain_p_l = torch.matmul(link_rot_g.transpose(-1, -2), terrain_delta_w[..., None]).squeeze(-1)

    effective_radii = radii_l.view(1, 1, 1, group_count, 1, 3) + float(cfg.collision_margin_m)
    delta_l = terrain_p_l - center_l[:, :, :, :, None, :]
    inside = (delta_l / effective_radii).square().sum(dim=-1) <= 1.0
    return torch.any(inside, dim=-1)


def ellipsoid_collision_mask(
    terrain: ParallelismTerrain,
    geometry,
    cfg: ParallelismCfg,
) -> tuple[Tensor, Tensor]:
    specs = tuple(cfg.collision_ellipsoids)
    batch, leg_count, candidate_count = geometry.foot_pos_w.shape[:3]
    collision_bits = torch.zeros(
        batch,
        leg_count,
        candidate_count,
        len(specs),
        dtype=torch.bool,
        device=geometry.foot_pos_w.device,
    )
    candidate_hit = torch.zeros(
        batch,
        leg_count,
        candidate_count,
        dtype=torch.bool,
        device=geometry.foot_pos_w.device,
    )
    for link_type in ("thigh", "calf", "foot"):
        indices = _group_indices(specs, link_type)
        if not indices:
            continue
        group_hit = _ellipsoid_group_collision(terrain, geometry, cfg, specs, indices)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=collision_bits.device)
        collision_bits.index_copy_(dim=-1, index=index_tensor, source=group_hit)
        candidate_hit = torch.logical_or(candidate_hit, torch.any(group_hit, dim=-1))
    return ~candidate_hit, collision_bits
