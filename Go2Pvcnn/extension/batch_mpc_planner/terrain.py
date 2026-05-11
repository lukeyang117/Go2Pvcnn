"""Terrain helpers for batch MPC planner."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .types import MpcPlannerTerrain


def _reshape_ray_hits(ray_hits_w: Tensor) -> Tensor:
    hits = torch.as_tensor(ray_hits_w)
    if hits.ndim == 2 and int(hits.shape[-1]) == 3:
        hits = hits.unsqueeze(0)
    if hits.ndim == 4 and int(hits.shape[-1]) == 3:
        return hits
    if hits.ndim == 3 and int(hits.shape[-1]) == 3:
        # [H, W, 3] (single env grid)
        if int(hits.shape[0]) > 1 and int(hits.shape[0]) == int(hits.shape[1]):
            return hits.unsqueeze(0)
        # [B, H*W, 3] (flattened scanner grid per env)
        ray_count = int(hits.shape[1])
        side = int(round(math.sqrt(ray_count)))
        if side * side != ray_count:
            raise ValueError(
                "ray_hits_w with shape [B, N, 3] requires square N=H*W, "
                f"got N={ray_count} for shape {tuple(hits.shape)}"
            )
        return hits.reshape(int(hits.shape[0]), side, side, 3)
    raise ValueError(
        "ray_hits_w must be one of [B,H,W,3], [B,H*W,3], [H,W,3], or [H*W,3], "
        f"got {tuple(hits.shape)}"
    )


def _reshape_semantic_map(
    semantic_map: Tensor | None,
    *,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor | None:
    if semantic_map is None:
        return None
    sem = torch.as_tensor(semantic_map, device=device)
    if sem.ndim == 1:
        if batch != 1 or int(sem.numel()) != height * width:
            raise ValueError(
                "semantic_map [H*W] requires single-env input and matching H*W; "
                f"got batch={batch}, semantic_map={tuple(sem.shape)}, target={(height, width)}"
            )
        sem = sem.reshape(1, height, width)
    elif sem.ndim == 2:
        if tuple(sem.shape) == (height, width):
            sem = sem.unsqueeze(0).expand(batch, -1, -1)
        elif tuple(sem.shape) == (batch, height * width):
            sem = sem.reshape(batch, height, width)
        else:
            raise ValueError(
                "semantic_map [2D] must be [H,W] or [B,H*W]; "
                f"got {tuple(sem.shape)} for target batch/grid {(batch, height, width)}"
            )
    elif sem.ndim == 3:
        if tuple(sem.shape) == (1, height, width) and batch > 1:
            sem = sem.expand(batch, -1, -1)
        elif tuple(sem.shape) != (batch, height, width):
            raise ValueError(
                "semantic_map [3D] must match [B,H,W]; "
                f"got {tuple(sem.shape)} for target {(batch, height, width)}"
            )
    else:
        raise ValueError(
            "semantic_map must be one of [B,H,W], [H,W], [B,H*W], or [H*W], "
            f"got {tuple(sem.shape)}"
        )
    if torch.is_floating_point(sem):
        sem = torch.nan_to_num(sem, nan=0.0, posinf=0.0, neginf=0.0)
    return sem.to(dtype=torch.long).contiguous()


def build_mpc_terrain_from_scanner(
    ray_hits_w: Tensor,
    *,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    semantic_map: Tensor | None = None,
) -> MpcPlannerTerrain:
    hits = torch.nan_to_num(_reshape_ray_hits(ray_hits_w), nan=0.0, posinf=0.0, neginf=0.0)
    height_map = hits[..., 2].to(dtype=torch.float32).contiguous()
    sem = _reshape_semantic_map(
        semantic_map,
        batch=int(height_map.shape[0]),
        height=int(height_map.shape[1]),
        width=int(height_map.shape[2]),
        device=height_map.device,
    )
    return MpcPlannerTerrain(
        height_map=height_map,
        semantic_map=sem,
        world_x_range=world_x_range,
        world_y_range=world_y_range,
    )


def subset_mpc_terrain(terrain: MpcPlannerTerrain, env_ids: Tensor) -> MpcPlannerTerrain:
    if terrain.height_map.ndim != 3:
        raise ValueError(f"terrain.height_map must be [B,H,W], got {tuple(terrain.height_map.shape)}")
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=terrain.height_map.device).reshape(-1)
    batch = int(terrain.height_map.shape[0])
    if int(ids.numel()) > 0:
        valid = torch.logical_and(ids >= 0, ids < batch)
        if not bool(torch.all(valid)):
            bad = ids[torch.logical_not(valid)]
            raise IndexError(
                f"env_ids out of bounds for terrain batch={batch}; "
                f"first bad ids={bad[:8].tolist()}"
            )
    height = terrain.height_map.index_select(0, ids)
    sem = terrain.semantic_map.index_select(0, ids) if terrain.semantic_map is not None else None
    return MpcPlannerTerrain(
        height_map=height,
        semantic_map=sem,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
    )


__all__ = ["build_mpc_terrain_from_scanner", "subset_mpc_terrain"]
