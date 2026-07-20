"""Build immutable terrain-field batches from aligned scanner rows."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiTerrainCfg
from extension.joint_mpc_rti.terrain.cost_map import build_soft_semantic_fields
from extension.joint_mpc_rti.terrain.distance_field import distance_gradient, semantic_mask, signed_boundary_distance
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
    terrain_cfg: JointMpcRtiTerrainCfg | None = None,
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
    if height.is_cuda:
        from extension.joint_mpc_rti.terrain.cuda_edt import semantic_distance_fields_cuda

        distance = semantic_distance_fields_cuda(
            semantic,
            small_ids=small_ids,
            large_ids=large_ids,
            resolution=resolution,
        )
        small_distance = distance[0]
        large_distance = distance[1]
        zero_gradient = distance.new_zeros((1, 1, 1, 2)).expand(batch, height.shape[1], height.shape[2], 2)
        small_gradient = zero_gradient
        large_gradient = zero_gradient
    else:
        small_mask = semantic_mask(semantic, small_ids)
        large_mask = semantic_mask(semantic, large_ids)
        small_distance = signed_boundary_distance(small_mask, resolution=resolution)
        large_distance = signed_boundary_distance(large_mask, resolution=resolution)
        small_gradient = distance_gradient(small_distance, resolution=resolution)
        large_gradient = distance_gradient(large_distance, resolution=resolution)
    soft = build_soft_semantic_fields(
        height,
        semantic,
        JointMpcRtiTerrainCfg() if terrain_cfg is None else terrain_cfg,
        resolution=resolution,
        small_ids=small_ids,
        large_ids=large_ids,
    )
    return JointMpcTerrainField(
        height_w=height.contiguous(),
        semantic_id=semantic.contiguous(),
        small_distance_m=small_distance.contiguous(),
        large_distance_m=large_distance.contiguous(),
        small_gradient_xy=small_gradient,
        large_gradient_xy=large_gradient,
        valid_mask=torch.isfinite(height).contiguous(),
        origin_w=origin.contiguous(),
        yaw_w=yaw.contiguous(),
        timestamp=stamp.contiguous(),
        version=field_version.contiguous(),
        resolution=float(resolution),
        small_occupancy=soft.small_occupancy[:, 0].contiguous(),
        large_occupancy=soft.large_occupancy[:, 0].contiguous(),
        small_propagated_height=soft.small_height[:, 0].contiguous(),
        large_propagated_height=soft.large_height[:, 0].contiguous(),
        small_occupancy_gradient_xy=soft.small_gradient_xy.permute(0, 2, 3, 1).contiguous(),
        large_occupancy_gradient_xy=soft.large_gradient_xy.permute(0, 2, 3, 1).contiguous(),
    )


__all__ = ["build_field_batch"]
