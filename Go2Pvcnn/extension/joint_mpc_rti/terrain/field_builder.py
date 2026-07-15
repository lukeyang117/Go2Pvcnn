"""Build immutable terrain-field batches from aligned scanner rows."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.terrain.distance_field import distance_gradient, jump_flood_distance, semantic_mask
from extension.joint_mpc_rti.types import JointMpcTerrainField


def build_field_batch(
    *,
    height_w: Tensor,
    semantic_id: Tensor,
    origin_w: Tensor,
    yaw_w: Tensor,
    timestamp: Tensor,
    version: Tensor,
    resolution: float,
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
) -> JointMpcTerrainField:
    height = torch.as_tensor(height_w, dtype=torch.float32)
    semantic = torch.as_tensor(semantic_id, dtype=torch.long, device=height.device)
    if height.ndim != 3 or semantic.shape != height.shape:
        raise ValueError("height_w and semantic_id must have matching shape [B,NX,NY]")
    batch = int(height.shape[0])
    origin = torch.as_tensor(origin_w, dtype=height.dtype, device=height.device)
    yaw = torch.as_tensor(yaw_w, dtype=height.dtype, device=height.device)
    stamp = torch.as_tensor(timestamp, dtype=height.dtype, device=height.device)
    field_version = torch.as_tensor(version, dtype=torch.long, device=height.device)
    if origin.shape != (batch, 3) or yaw.shape != (batch,) or stamp.shape != (batch,) or field_version.shape != (batch,):
        raise ValueError("field pose, timestamp, and version shapes must match the batch")
    small_distance = jump_flood_distance(semantic_mask(semantic, small_ids), resolution=resolution)
    large_distance = jump_flood_distance(semantic_mask(semantic, large_ids), resolution=resolution)
    return JointMpcTerrainField(
        height_w=height.contiguous(),
        semantic_id=semantic.contiguous(),
        small_distance_m=small_distance.contiguous(),
        large_distance_m=large_distance.contiguous(),
        small_gradient_xy=distance_gradient(small_distance, resolution=resolution).contiguous(),
        large_gradient_xy=distance_gradient(large_distance, resolution=resolution).contiguous(),
        valid_mask=torch.isfinite(height).contiguous(),
        origin_w=origin.contiguous(),
        yaw_w=yaw.contiguous(),
        timestamp=stamp.contiguous(),
        version=field_version.contiguous(),
        resolution=float(resolution),
    )


__all__ = ["build_field_batch"]
