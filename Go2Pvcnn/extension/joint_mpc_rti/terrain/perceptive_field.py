"""Current-refresh fixed-channel 2.5D perceptive field."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg, JointMpcRtiTerrainCfg
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.types import JointMpcFieldFrame, JointMpcPerceptiveField


GEOMETRY_NAMES = ("foot", "knee", "calf", "thigh", "base")


def _semantic_mask(semantic_id: Tensor, ids: tuple[int, ...]) -> Tensor:
    mask = torch.zeros_like(semantic_id, dtype=torch.bool)
    for semantic_id_value in ids:
        mask |= semantic_id == int(semantic_id_value)
    return mask


def _kernel_radius(radius_m: float, resolution: float) -> int:
    return int(math.ceil(float(radius_m) / float(resolution)))


def _inflate_mask(mask: Tensor, radius_m: float, resolution: float) -> Tensor:
    radius = _kernel_radius(radius_m, resolution)
    if radius == 0:
        return mask
    padded = F.pad(mask[:, None].to(torch.float32), (radius,) * 4, value=1.0)
    return F.max_pool2d(padded, 2 * radius + 1, stride=1)[:, 0] > 0.0


def _pool_height(height: Tensor, radius_m: float, resolution: float, wall: float) -> Tensor:
    radius = _kernel_radius(radius_m, resolution)
    if radius == 0:
        return height
    padded = F.pad(height[:, None], (radius,) * 4, value=float(wall))
    return F.max_pool2d(padded, 2 * radius + 1, stride=1)[:, 0]


def _slope_and_roughness(
    height: Tensor,
    *,
    resolution: float,
    roughness_radius_m: float,
) -> tuple[Tensor, Tensor, Tensor]:
    scharr_x = constant_like(
        height,
        "perceptive_scharr_x",
        ((-3.0, -10.0, -3.0), (0.0, 0.0, 0.0), (3.0, 10.0, 3.0)),
    ).view(1, 1, 3, 3)
    scharr_y = constant_like(
        height,
        "perceptive_scharr_y",
        ((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0)),
    ).view(1, 1, 3, 3)
    padded = F.pad(height[:, None], (1, 1, 1, 1), mode="replicate")
    scale = 1.0 / (32.0 * float(resolution))
    slope_x = F.conv2d(padded, scharr_x)[:, 0] * scale
    slope_y = F.conv2d(padded, scharr_y)[:, 0] * scale
    slope_xy = torch.stack((slope_x, slope_y), dim=-1)
    slope_rad = torch.atan(torch.linalg.vector_norm(slope_xy, dim=-1))

    radius = _kernel_radius(roughness_radius_m, resolution)
    if radius == 0:
        roughness = torch.zeros_like(height)
    else:
        local = F.pad(height[:, None], (radius,) * 4, mode="replicate")
        maximum = F.max_pool2d(local, 2 * radius + 1, stride=1)[:, 0]
        minimum = -F.max_pool2d(-local, 2 * radius + 1, stride=1)[:, 0]
        roughness = maximum - minimum
    return slope_xy, slope_rad, roughness


def validate_frame_freshness(
    *,
    field_refresh_id: Tensor,
    state_refresh_id: Tensor,
    field_timestamp: Tensor | None = None,
    state_timestamp: Tensor | None = None,
) -> Tensor:
    field_id = torch.as_tensor(field_refresh_id)
    state_id = torch.as_tensor(state_refresh_id, dtype=field_id.dtype, device=field_id.device)
    fresh = field_id == state_id
    if field_timestamp is not None or state_timestamp is not None:
        if field_timestamp is None or state_timestamp is None:
            return torch.zeros_like(fresh, dtype=torch.bool)
        field_time = torch.as_tensor(field_timestamp)
        state_time = torch.as_tensor(
            state_timestamp, dtype=field_time.dtype, device=field_time.device
        )
        fresh = fresh & (field_time == state_time)
    return fresh


def build_perceptive_field(
    height_w: Tensor,
    semantic_id: Tensor,
    valid_mask: Tensor,
    frame: JointMpcFieldFrame,
    cfg: JointMpcRtiCfg | JointMpcRtiTerrainCfg,
) -> JointMpcPerceptiveField:
    terrain = cfg.terrain if isinstance(cfg, JointMpcRtiCfg) else cfg
    height = torch.as_tensor(height_w, dtype=torch.float32)
    semantic = torch.as_tensor(semantic_id, dtype=torch.long, device=height.device)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=height.device)
    if height.ndim != 3 or semantic.shape != height.shape or valid.shape != height.shape:
        raise ValueError("height_w, semantic_id, and valid_mask must have shape [B,NX,NY]")
    if int(frame.origin_w.shape[0]) != int(height.shape[0]):
        raise ValueError("frame metadata batch must match the field batch")

    resolution = float(terrain.resolution)
    finite = torch.isfinite(height)
    valid = valid & finite
    small = _semantic_mask(semantic, tuple(terrain.small_ids)) & valid
    large = _semantic_mask(semantic, tuple(terrain.large_ids)) & valid
    unknown = ~valid
    wall = float(terrain.h_wall)
    physical_height = torch.where(finite, height, torch.zeros_like(height))
    effective_height = torch.where(large | unknown, height.new_full((), wall), physical_height)

    foot_height = _pool_height(
        effective_height,
        float(terrain.foot_radius_m + terrain.foot_margin_m),
        resolution,
        wall,
    )
    knee_height = _pool_height(
        effective_height,
        float(terrain.knee_radius_m + terrain.link_margin_m),
        resolution,
        wall,
    )
    calf_height = _pool_height(
        effective_height,
        float(terrain.calf_radius_m + terrain.link_margin_m),
        resolution,
        wall,
    )
    thigh_height = _pool_height(
        effective_height,
        float(terrain.thigh_radius_m + terrain.link_margin_m),
        resolution,
        wall,
    )
    base_height = _pool_height(
        effective_height,
        float(terrain.base_radius_m + terrain.base_margin_m),
        resolution,
        wall,
    )
    inflated_height = torch.stack(
        (foot_height, knee_height, calf_height, thigh_height, base_height), dim=1
    )

    slope_xy, slope_rad, roughness = _slope_and_roughness(
        physical_height,
        resolution=resolution,
        roughness_radius_m=float(terrain.roughness_radius_m),
    )
    semantic_edge = (slope_rad > float(terrain.slope_max_rad)) | (
        roughness > float(terrain.roughness_max_m)
    )
    landing_forbidden = _inflate_mask(
        small | large | unknown,
        float(terrain.foot_radius_m + terrain.landing_margin_m),
        resolution,
    )
    edge_forbidden = _inflate_mask(
        semantic_edge,
        float(terrain.edge_margin_m),
        resolution,
    )
    landing_safe = valid & ~landing_forbidden & ~edge_forbidden

    return JointMpcPerceptiveField(
        height_w=physical_height.contiguous(),
        semantic_id=semantic.contiguous(),
        valid_mask=valid.contiguous(),
        small_mask=small.contiguous(),
        large_mask=large.contiguous(),
        unknown_mask=unknown.contiguous(),
        inflated_height_w=inflated_height.contiguous(),
        landing_safe=landing_safe.contiguous(),
        slope_xy=slope_xy.contiguous(),
        slope_rad=slope_rad.contiguous(),
        roughness=roughness.contiguous(),
        semantic_edge_mask=semantic_edge.contiguous(),
        origin_w=frame.origin_w.to(dtype=height.dtype, device=height.device).contiguous(),
        yaw_w=frame.yaw_w.to(dtype=height.dtype, device=height.device).contiguous(),
        timestamp=frame.timestamp.to(dtype=height.dtype, device=height.device).contiguous(),
        refresh_id=frame.refresh_id.to(dtype=torch.long, device=height.device).contiguous(),
        resolution=resolution,
    )


__all__ = [
    "GEOMETRY_NAMES",
    "build_perceptive_field",
    "validate_frame_freshness",
]
