"""GPU-resident terrain tensors and batched height queries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TogetherPlannerConfig
from .types import (
    TOGETHER_TERRAIN_REFERENCE_STENCIL_RADIUS_M,
    TOGETHER_TERRAIN_REFERENCE_STENCIL_STEP_M,
    TogetherTerrainSemanticId,
)


def _ensure_heightmap(heightmap: Tensor) -> Tensor:
    maps = torch.as_tensor(heightmap)
    if maps.ndim == 2:
        maps = maps.unsqueeze(0).unsqueeze(1)
    elif maps.ndim == 3:
        maps = maps.unsqueeze(1)
    if maps.ndim != 4 or maps.shape[1] != 1:
        raise ValueError("heightmap must have shape [H, W], [B, H, W], or [B, 1, H, W]")
    if not torch.is_floating_point(maps):
        maps = maps.to(torch.float32)
    return torch.nan_to_num(maps.contiguous(), nan=0.0, posinf=0.0, neginf=0.0)


def _ensure_semantic_map(semantic_map: Tensor | None, heightmaps: Tensor) -> Tensor | None:
    if semantic_map is None:
        return None
    semantic = torch.as_tensor(semantic_map)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0).unsqueeze(0)
    elif semantic.ndim == 3:
        semantic = semantic.unsqueeze(1)
    if semantic.ndim != 4 or semantic.shape[1] != 1:
        raise ValueError("semantic_map must have shape [H, W], [B, H, W], or [B, 1, H, W]")
    if semantic.shape != heightmaps.shape:
        raise ValueError("semantic_map must match heightmap batch, channel, height, and width")
    if semantic.device != heightmaps.device:
        raise ValueError("semantic_map must be on the same device as heightmap")
    return torch.nan_to_num(semantic.to(dtype=torch.long).contiguous(), nan=0.0, posinf=0.0, neginf=0.0)


def _reshape_ray_hits(ray_hits: Tensor) -> Tensor:
    hits = torch.as_tensor(ray_hits)
    if hits.ndim == 4 and hits.shape[-1] == 3:
        return hits
    if hits.ndim == 3 and hits.shape[-1] == 3:
        ray_count = int(hits.shape[1])
        side = int(round(math.sqrt(ray_count)))
        if side * side != ray_count:
            raise ValueError("ray hit count must be square")
        return hits.reshape(hits.shape[0], side, side, 3)
    raise ValueError("ray_hits must have shape [B, H, W, 3] or [B, H*W, 3]")


def _normalize_points(points_xy: Tensor, x_range: tuple[float, float], y_range: tuple[float, float]) -> Tensor:
    x0, x1 = x_range
    y0, y1 = y_range
    xs = points_xy[..., 0].clamp(float(x0), float(x1))
    ys = points_xy[..., 1].clamp(float(y0), float(y1))
    x_norm = (xs - float(x0)) / max(float(x1) - float(x0), 1e-9) * 2.0 - 1.0
    y_norm = (float(y1) - ys) / max(float(y1) - float(y0), 1e-9) * 2.0 - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def _fixed_reference_offsets(device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = TOGETHER_TERRAIN_REFERENCE_STENCIL_RADIUS_M
    step = TOGETHER_TERRAIN_REFERENCE_STENCIL_STEP_M
    axis = torch.arange(-radius, radius + 0.5 * step, step, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)


def _semantic_id_tensor(semantic_id: int | TogetherTerrainSemanticId, device: torch.device) -> Tensor:
    return torch.as_tensor(semantic_id, device=device, dtype=torch.long)


def _sample_heightmap(
    heightmaps: Tensor,
    points_xy: Tensor,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    mode: str,
) -> Tensor:
    grid = _normalize_points(points_xy, x_range, y_range).unsqueeze(2)
    sampled = F.grid_sample(
        heightmaps,
        grid,
        mode=mode,
        align_corners=True,
        padding_mode="border",
    )
    return sampled[:, 0, :, 0]


@dataclass(frozen=True)
class TogetherPlannerTerrain:
    heightmaps: Tensor
    world_x_range: tuple[float, float]
    world_y_range: tuple[float, float]
    semantic_maps: Tensor | None = None

    @classmethod
    def from_heightmap(
        cls,
        heightmap: Tensor,
        *,
        world_x_range: tuple[float, float],
        world_y_range: tuple[float, float],
        semantic_map: Tensor | None = None,
    ) -> "TogetherPlannerTerrain":
        heightmaps = _ensure_heightmap(heightmap)
        return cls(
            heightmaps=heightmaps,
            world_x_range=(float(world_x_range[0]), float(world_x_range[1])),
            world_y_range=(float(world_y_range[0]), float(world_y_range[1])),
            semantic_maps=_ensure_semantic_map(semantic_map, heightmaps),
        )

    @classmethod
    def from_ray_hits(
        cls,
        ray_hits: Tensor,
        *,
        world_x_range: tuple[float, float],
        world_y_range: tuple[float, float],
        semantic_map: Tensor | None = None,
    ) -> "TogetherPlannerTerrain":
        hits = torch.nan_to_num(_reshape_ray_hits(ray_hits), nan=0.0, posinf=0.0, neginf=0.0)
        return cls.from_heightmap(
            hits[..., 2],
            world_x_range=world_x_range,
            world_y_range=world_y_range,
            semantic_map=semantic_map,
        )

    @property
    def batch_size(self) -> int:
        return int(self.heightmaps.shape[0])

    @property
    def device(self) -> torch.device:
        return self.heightmaps.device

    @property
    def dtype(self) -> torch.dtype:
        return self.heightmaps.dtype

    def _batched_points(self, points_xy: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        points = torch.as_tensor(points_xy, device=self.device, dtype=self.dtype)
        if points.ndim == 2 and points.shape[-1] == 2:
            points = points.unsqueeze(0).expand(self.batch_size, -1, -1)
        if points.ndim < 3 or points.shape[0] != self.batch_size or points.shape[-1] != 2:
            raise ValueError("points_xy must have shape [B, ..., 2]")
        query_shape = tuple(points.shape[1:-1])
        return points.reshape(self.batch_size, -1, 2), query_shape

    def height_at(self, points_xy: Tensor) -> Tensor:
        points, query_shape = self._batched_points(points_xy)
        sampled = _sample_heightmap(
            self.heightmaps,
            points,
            x_range=self.world_x_range,
            y_range=self.world_y_range,
            mode="bilinear",
        )
        return sampled.reshape(self.batch_size, *query_shape)

    def semantic_at(self, points_xy: Tensor) -> Tensor:
        points, query_shape = self._batched_points(points_xy)
        if self.semantic_maps is None:
            return torch.full(
                (self.batch_size, *query_shape),
                TogetherTerrainSemanticId.TERRAIN.value,
                device=self.device,
                dtype=torch.long,
            )
        grid = _normalize_points(points, self.world_x_range, self.world_y_range).unsqueeze(2)
        sampled = F.grid_sample(
            self.semantic_maps.to(dtype=self.dtype),
            grid,
            mode="nearest",
            align_corners=True,
            padding_mode="border",
        )
        return sampled[:, 0, :, 0].reshape(self.batch_size, *query_shape).to(dtype=torch.long)

    def obstacle_mask_at(self, points_xy: Tensor, semantic_id: int | TogetherTerrainSemanticId) -> Tensor:
        return self.semantic_at(points_xy) == _semantic_id_tensor(semantic_id, self.device)

    def obstacle_height_at(self, points_xy: Tensor, semantic_id: int | TogetherTerrainSemanticId) -> Tensor:
        heights = self.height_at(points_xy)
        return torch.where(
            self.obstacle_mask_at(points_xy, semantic_id),
            heights,
            torch.zeros_like(heights),
        )

    def terrain_reference_height_at(self, points_xy: Tensor) -> Tensor:
        points, query_shape = self._batched_points(points_xy)
        offsets = _fixed_reference_offsets(self.device, self.dtype)
        candidates = points[:, :, None, :] + offsets.view(1, 1, -1, 2)
        flat_candidates = candidates.reshape(self.batch_size, -1, 2)
        sampled_height = _sample_heightmap(
            self.heightmaps,
            flat_candidates,
            x_range=self.world_x_range,
            y_range=self.world_y_range,
            mode="nearest",
        ).reshape(self.batch_size, points.shape[1], offsets.shape[0])
        if self.semantic_maps is None:
            return torch.amin(sampled_height, dim=-1).reshape(self.batch_size, *query_shape)
        sampled_semantic = self.semantic_at(flat_candidates).reshape(self.batch_size, points.shape[1], offsets.shape[0])
        terrain_mask = sampled_semantic == _semantic_id_tensor(TogetherTerrainSemanticId.TERRAIN, self.device)
        inf_fill = torch.full_like(sampled_height, float("inf"))
        terrain_heights = torch.where(terrain_mask, sampled_height, inf_fill)
        min_height = torch.amin(terrain_heights, dim=-1)
        query_height = _sample_heightmap(
            self.heightmaps,
            points,
            x_range=self.world_x_range,
            y_range=self.world_y_range,
            mode="nearest",
        ).reshape(self.batch_size, points.shape[1])
        finite_mask = torch.isfinite(min_height)
        resolved = torch.where(finite_mask, min_height, query_height)
        return resolved.reshape(self.batch_size, *query_shape)

    def obstacle_relative_height_at(self, points_xy: Tensor, semantic_id: int | TogetherTerrainSemanticId) -> Tensor:
        return self.obstacle_height_at(points_xy, semantic_id) - self.terrain_reference_height_at(points_xy)

    def slope_at(self, points_xy: Tensor, cfg: TogetherPlannerConfig | None = None) -> Tensor:
        planner_cfg = cfg or TogetherPlannerConfig()
        points = torch.as_tensor(points_xy, device=self.device, dtype=self.dtype)
        step = torch.as_tensor(float(planner_cfg.slope_sample_step), device=self.device, dtype=self.dtype)
        offset_x = torch.stack((step, torch.zeros_like(step)), dim=0)
        offset_y = torch.stack((torch.zeros_like(step), step), dim=0)
        hx0 = self.height_at(points - offset_x)
        hx1 = self.height_at(points + offset_x)
        hy0 = self.height_at(points - offset_y)
        hy1 = self.height_at(points + offset_y)
        dzdx = (hx1 - hx0) / (2.0 * step)
        dzdy = (hy1 - hy0) / (2.0 * step)
        return torch.sqrt(dzdx * dzdx + dzdy * dzdy)

    def support_at(
        self,
        points_xy: Tensor,
        cfg: TogetherPlannerConfig | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        planner_cfg = cfg or TogetherPlannerConfig()
        points, query_shape = self._batched_points(points_xy)
        step = float(planner_cfg.support_search_step)
        radius = float(planner_cfg.support_search_radius)
        search_count = max(1, int(math.ceil(radius / max(step, 1e-6))))
        axis = torch.arange(-search_count, search_count + 1, device=self.device, dtype=self.dtype) * step
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        candidates = points[:, :, None, :] + offsets.view(1, 1, -1, 2)
        flat_candidates = candidates.reshape(self.batch_size, -1, 2)
        candidate_height = self.height_at(flat_candidates).reshape(self.batch_size, points.shape[1], offsets.shape[0])
        candidate_slope = self.slope_at(flat_candidates, planner_cfg).reshape(self.batch_size, points.shape[1], offsets.shape[0])
        candidate_semantic = self.semantic_at(flat_candidates).reshape(self.batch_size, points.shape[1], offsets.shape[0])
        terrain_semantic = _semantic_id_tensor(TogetherTerrainSemanticId.TERRAIN, self.device)
        legal_support = candidate_semantic == terrain_semantic
        distance = torch.linalg.vector_norm(offsets, dim=-1).view(1, 1, -1)
        walk_penalty = torch.relu(candidate_slope - float(planner_cfg.support_walkable_slope))
        score = walk_penalty * 2.0 + distance * 0.25 - candidate_height * 0.05
        score = torch.where(legal_support, score, torch.full_like(score, float("inf")))
        has_legal_support = legal_support.any(dim=-1)
        best = torch.argmin(score, dim=-1)
        best_xy = candidates.gather(2, best[:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)
        best_height = candidate_height.gather(2, best[:, :, None]).squeeze(2)
        best_slope = candidate_slope.gather(2, best[:, :, None]).squeeze(2)
        invalid_xy = torch.full_like(best_xy, float("nan"))
        invalid_scalar = torch.full_like(best_height, float("nan"))
        best_xy = torch.where(has_legal_support.unsqueeze(-1), best_xy, invalid_xy)
        best_height = torch.where(has_legal_support, best_height, invalid_scalar)
        best_slope = torch.where(has_legal_support, best_slope, invalid_scalar)
        return (
            best_xy.reshape(self.batch_size, *query_shape, 2),
            best_height.reshape(self.batch_size, *query_shape),
            best_slope.reshape(self.batch_size, *query_shape),
        )


__all__ = ["TogetherPlannerTerrain"]


def build_together_terrain_from_scanner(
    ray_hits: Tensor,
    *,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    semantic_map: Tensor | None = None,
) -> TogetherPlannerTerrain:
    return TogetherPlannerTerrain.from_ray_hits(
        ray_hits,
        world_x_range=world_x_range,
        world_y_range=world_y_range,
        semantic_map=semantic_map,
    )


__all__ = ["TogetherPlannerTerrain", "build_together_terrain_from_scanner"]
