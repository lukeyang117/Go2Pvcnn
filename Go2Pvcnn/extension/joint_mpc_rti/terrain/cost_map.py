"""Convolutional elevation-semantic fields used by trajectory optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg, JointMpcRtiTerrainCfg


@dataclass(frozen=True)
class SoftSemanticFields:
    occupancy: Tensor
    propagated_height: Tensor
    gradient_xy: Tensor

    @property
    def small_occupancy(self) -> Tensor:
        return self.occupancy[:, 0:1]

    @property
    def large_occupancy(self) -> Tensor:
        return self.occupancy[:, 1:2]

    @property
    def small_height(self) -> Tensor:
        return self.propagated_height[:, 0:1]

    @property
    def large_height(self) -> Tensor:
        return self.propagated_height[:, 1:2]

    @property
    def small_gradient_xy(self) -> Tensor:
        return self.gradient_xy[:, 0]

    @property
    def large_gradient_xy(self) -> Tensor:
        return self.gradient_xy[:, 1]


@dataclass(frozen=True)
class EffectiveSurface:
    height_w: Tensor
    occupancy: Tensor


def _class_mask(semantic_id: Tensor, ids: tuple[int, ...]) -> Tensor:
    values = torch.as_tensor(ids, dtype=semantic_id.dtype, device=semantic_id.device)
    return (semantic_id[..., None] == values).any(dim=-1)


def _gaussian_kernel(
    sigma_m: float,
    radius: int,
    resolution: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    coordinate = torch.arange(-radius, radius + 1, dtype=dtype, device=device) * float(resolution)
    distance_sq = coordinate[:, None].square() + coordinate[None, :].square()
    return torch.exp(-0.5 * distance_sq / float(sigma_m) ** 2)


def build_soft_semantic_fields(
    height_w: Tensor,
    semantic_id: Tensor,
    cfg: JointMpcRtiTerrainCfg,
    *,
    resolution: float,
    small_ids: tuple[int, ...] | None = None,
    large_ids: tuple[int, ...] | None = None,
) -> SoftSemanticFields:
    """Build two soft occupancy/height channels with fixed grouped convolutions."""
    height = torch.as_tensor(height_w)
    semantic = torch.as_tensor(semantic_id, dtype=torch.long, device=height.device)
    if height.ndim != 3 or semantic.shape != height.shape:
        raise ValueError("height_w and semantic_id must have shape [B,H,W]")
    radius = int(cfg.kernel_radius_cells)
    if radius < 1:
        raise ValueError("kernel_radius_cells must be positive")
    small_values = cfg.small_ids if small_ids is None else small_ids
    large_values = cfg.large_ids if large_ids is None else large_ids
    mask = torch.stack(
        (_class_mask(semantic, small_values), _class_mask(semantic, large_values)),
        dim=1,
    ).to(height.dtype)
    weighted_height = mask * height[:, None]
    convolution_input = torch.cat((mask, weighted_height), dim=1)
    small_kernel = _gaussian_kernel(
        cfg.small_sigma_m, radius, resolution, dtype=height.dtype, device=height.device
    )
    large_kernel = _gaussian_kernel(
        cfg.large_sigma_m, radius, resolution, dtype=height.dtype, device=height.device
    )
    grouped_kernels = torch.stack(
        (small_kernel, large_kernel, small_kernel, large_kernel),
        dim=0,
    )[:, None]
    convolved = F.conv2d(convolution_input, grouped_kernels, padding=radius, groups=4)
    mass = convolved[:, :2]
    height_numerator = convolved[:, 2:]
    gain = height.new_tensor((cfg.small_gain, cfg.large_gain)).view(1, 2, 1, 1)
    occupancy = 1.0 - torch.exp(-gain * mass)
    propagated_height = height_numerator / mass.clamp_min(1.0e-6)

    scharr_x = height.new_tensor(((-3.0, -10.0, -3.0), (0.0, 0.0, 0.0), (3.0, 10.0, 3.0)))
    scharr_y = height.new_tensor(((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0)))
    scharr = torch.stack((scharr_x, scharr_y, scharr_x, scharr_y), dim=0)[:, None]
    scharr = scharr / (32.0 * float(resolution))
    gradient_xy = F.conv2d(occupancy, scharr, padding=1, groups=2).reshape(
        height.shape[0], 2, 2, height.shape[1], height.shape[2]
    )
    return SoftSemanticFields(
        occupancy=occupancy,
        propagated_height=propagated_height,
        gradient_xy=gradient_xy,
    )


def effective_surface(
    query,
    *,
    body_part: str,
    stance: bool,
    cfg: JointMpcRtiCfg,
) -> EffectiveSurface:
    """Blend real elevation, propagated class height, and continuous virtual walls."""
    if body_part not in {"foot", "knee", "calf", "thigh", "base"}:
        raise ValueError(f"unknown body part: {body_part}")
    small_target = (
        torch.full_like(query.height_w, float(cfg.terrain.h_wall))
        if stance and body_part == "foot"
        else query.small_propagated_height
    )
    small_height = torch.lerp(query.height_w, small_target, query.small_occupancy)
    wall = torch.full_like(small_height, float(cfg.terrain.h_wall))
    height = torch.lerp(small_height, wall, query.large_occupancy)
    occupancy = 1.0 - (1.0 - query.small_occupancy) * (1.0 - query.large_occupancy)
    return EffectiveSurface(height_w=height, occupancy=occupancy)


__all__ = [
    "EffectiveSurface",
    "SoftSemanticFields",
    "build_soft_semantic_fields",
    "effective_surface",
]
