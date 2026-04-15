"""Batched GPU terrain queries for planner heightmaps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_ranges(
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
) -> None:
    x0, x1 = world_x_range
    y0, y1 = world_y_range
    if not x0 < x1:
        raise ValueError("world_x_range must be increasing")
    if not y0 < y1:
        raise ValueError("world_y_range must be increasing")


def _normalize_points(
    points_xy: Tensor,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
) -> Tensor:
    x0, x1 = world_x_range
    y0, y1 = world_y_range
    xs = points_xy[..., 0].clamp(x0, x1)
    ys = points_xy[..., 1].clamp(y0, y1)

    if x1 == x0:
        raise ValueError("world_x_range must span a nonzero width")
    if y1 == y0:
        raise ValueError("world_y_range must span a nonzero height")

    x_norm = (xs - x0) / (x1 - x0) * 2.0 - 1.0
    # Grid-sample uses top-to-bottom image coordinates; larger world-y values map upward.
    y_norm = (y1 - ys) / (y1 - y0) * 2.0 - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def _ensure_heightmaps(heightmaps: Tensor) -> Tensor:
    heightmaps = torch.as_tensor(heightmaps)
    if heightmaps.ndim == 3:
        heightmaps = heightmaps.unsqueeze(1)
    if heightmaps.ndim != 4:
        raise ValueError(f"heightmaps must be 3D or 4D, got {tuple(heightmaps.shape)}")
    if heightmaps.shape[1] != 1:
        raise ValueError("heightmaps must have a single channel with shape (N, 1, H, W)")
    if not torch.is_floating_point(heightmaps):
        heightmaps = heightmaps.to(torch.float32)
    return heightmaps.contiguous()


def _sanitize_ray_hits(ray_hits: Tensor) -> Tensor:
    hits = torch.as_tensor(ray_hits)
    if not torch.is_floating_point(hits):
        hits = hits.to(torch.float32)
    return torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0)


def _reshape_ray_hits(ray_hits: Tensor) -> Tensor:
    hits = torch.as_tensor(ray_hits)
    if hits.ndim == 4:
        if hits.shape[-1] != 3:
            raise ValueError("ray_hits must have shape (N, H, W, 3)")
        return hits
    if hits.ndim == 3 and hits.shape[-1] == 3:
        ray_count = int(hits.shape[1])
        side = int(round(math.sqrt(ray_count)))
        if side * side != ray_count:
            raise ValueError(f"ray count {ray_count} is not a perfect square")
        return hits.reshape(hits.shape[0], side, side, 3)
    if hits.ndim == 2 and hits.shape[-1] == 3:
        ray_count = int(hits.shape[0])
        side = int(round(math.sqrt(ray_count)))
        if side * side != ray_count:
            raise ValueError(f"ray count {ray_count} is not a perfect square")
        return hits.reshape(1, side, side, 3)
    raise ValueError("ray_hits must have shape (N, H, W, 3), (N, H*W, 3), or (H*W, 3)")


def _heightmaps_from_ray_hits(ray_hits: Tensor) -> Tensor:
    hits = _sanitize_ray_hits(_reshape_ray_hits(ray_hits))
    return hits[..., 2].unsqueeze(1).contiguous()


def _canonical_world_ranges_from_ray_hits(ray_hits: Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    hits = torch.as_tensor(_reshape_ray_hits(ray_hits), dtype=torch.float64)
    xy = hits[..., :2]
    finite_mask = torch.isfinite(xy).all(dim=-1)
    if not torch.any(finite_mask):
        raise ValueError("ray_hits must contain at least one finite x/y coordinate")

    finite_xy = xy[finite_mask]
    world_x_range = (float(finite_xy[:, 0].amin().item()), float(finite_xy[:, 0].amax().item()))
    world_y_range = (float(finite_xy[:, 1].amin().item()), float(finite_xy[:, 1].amax().item()))
    _validate_ranges(world_x_range, world_y_range)
    return world_x_range, world_y_range


def _reshape_points(points_xy: Tensor, batch_size: int) -> tuple[Tensor, bool]:
    if points_xy.ndim == 1:
        if points_xy.shape[0] != 2:
            raise ValueError("points_xy must have last dimension 2")
        return points_xy.view(1, 1, 2), True

    if points_xy.ndim == 2:
        if points_xy.shape[-1] != 2:
            raise ValueError("points_xy must have last dimension 2")
        if batch_size != 1:
            raise ValueError("points_xy with shape (P, 2) is only supported for a single terrain; use (N, P, 2) for batched queries")
        return points_xy.unsqueeze(0), True

    if points_xy.ndim == 3 and points_xy.shape[-1] == 2:
        if points_xy.shape[0] != batch_size:
            raise ValueError("batched points must match terrain batch size")
        return points_xy, False

    raise ValueError(f"points_xy must have shape (2,), (P, 2), or (N, P, 2); got {tuple(points_xy.shape)}")


def _reshape_endpoints(point_xy: Tensor, batch_size: int) -> tuple[Tensor, bool]:
    if point_xy.ndim == 1:
        if point_xy.shape[0] != 2:
            raise ValueError("segment endpoints must have last dimension 2")
        return point_xy.view(1, 2), True

    if point_xy.ndim == 2 and point_xy.shape[-1] == 2:
        if point_xy.shape[0] != batch_size:
            raise ValueError("batched segment endpoints must match terrain batch size")
        return point_xy, False

    raise ValueError(f"segment endpoints must have shape (2,) or (N, 2); got {tuple(point_xy.shape)}")


class BatchedTerrain:
    """Batched heightmap terrain with bilinear sampling."""

    def __init__(
        self,
        heightmaps: Tensor,
        *,
        world_x_range: tuple[float, float],
        world_y_range: tuple[float, float],
    ) -> None:
        _validate_ranges(world_x_range, world_y_range)
        self.world_x_range = (float(world_x_range[0]), float(world_x_range[1]))
        self.world_y_range = (float(world_y_range[0]), float(world_y_range[1]))
        self.heightmaps = _ensure_heightmaps(torch.as_tensor(heightmaps))
        if self.heightmaps.shape[-2] < 1 or self.heightmaps.shape[-1] < 1:
            raise ValueError("heightmaps must be non-empty")

    @classmethod
    def from_ray_hits(
        cls,
        ray_hits: Tensor,
        *,
        world_x_range: tuple[float, float],
        world_y_range: tuple[float, float],
    ) -> "BatchedTerrain":
        return cls(
            _heightmaps_from_ray_hits(ray_hits),
            world_x_range=world_x_range,
            world_y_range=world_y_range,
        )

    @property
    def batch_size(self) -> int:
        return int(self.heightmaps.shape[0])

    @property
    def height(self) -> int:
        return int(self.heightmaps.shape[-2])

    @property
    def width(self) -> int:
        return int(self.heightmaps.shape[-1])

    def _step_scale(self) -> float:
        x0, x1 = self.world_x_range
        x_step = (x1 - x0) / max(self.width - 1, 1)
        return max(x_step, 1e-9)

    def _sample_map(self, grid: Tensor, points_xy: Tensor) -> Tensor:
        points_xy = torch.as_tensor(points_xy, device=grid.device, dtype=grid.dtype)
        batch_size = int(grid.shape[0])
        points_xy, squeezed_single_point = _reshape_points(points_xy, batch_size)

        if points_xy.shape[0] == 1 and batch_size > 1:
            points_xy = points_xy.expand(batch_size, -1, -1)

        if points_xy.shape[0] != batch_size:
            raise ValueError("point batch size must match grid batch size")

        grid_xy = _normalize_points(points_xy, self.world_x_range, self.world_y_range)
        sample_grid = grid_xy.unsqueeze(2)
        sampled = F.grid_sample(
            grid,
            sample_grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        sampled = sampled[:, 0, :, 0]

        if squeezed_single_point and sampled.shape[1] == 1:
            return sampled[:, 0]
        return sampled

    def height_at(self, points_xy: Tensor) -> Tensor:
        return self._sample_map(self.heightmaps, points_xy)

    def roughness_at(self, points_xy: Tensor) -> Tensor:
        points_xy = torch.as_tensor(points_xy, device=self.heightmaps.device, dtype=self.heightmaps.dtype)
        points_xy, squeezed_single_point = _reshape_points(points_xy, self.batch_size)

        if points_xy.shape[0] == 1 and self.batch_size > 1:
            points_xy = points_xy.expand(self.batch_size, -1, -1)

        if points_xy.shape[0] != self.batch_size:
            raise ValueError("point batch size must match terrain batch size")

        x0, x1 = self.world_x_range
        y0, y1 = self.world_y_range
        dx = (x1 - x0) / max(self.width - 1, 1)
        dy = (y1 - y0) / max(self.height - 1, 1)
        if dx == 0.0 and dy == 0.0:
            roughness = torch.zeros(points_xy.shape[:-1], device=self.heightmaps.device, dtype=self.heightmaps.dtype)
            return roughness[:, 0] if squeezed_single_point and roughness.ndim == 2 and roughness.shape[1] == 1 else roughness

        left = self._sample_map(self.heightmaps, torch.stack((points_xy[..., 0] - dx, points_xy[..., 1]), dim=-1))
        right = self._sample_map(self.heightmaps, torch.stack((points_xy[..., 0] + dx, points_xy[..., 1]), dim=-1))
        down = self._sample_map(self.heightmaps, torch.stack((points_xy[..., 0], points_xy[..., 1] - dy), dim=-1))
        up = self._sample_map(self.heightmaps, torch.stack((points_xy[..., 0], points_xy[..., 1] + dy), dim=-1))

        dzdx = torch.zeros_like(left)
        dzdy = torch.zeros_like(left)
        if dx != 0.0:
            dzdx = (right - left) / (2.0 * dx)
        if dy != 0.0:
            dzdy = (up - down) / (2.0 * dy)
        roughness = torch.sqrt(dzdx * dzdx + dzdy * dzdy)
        if squeezed_single_point and roughness.ndim == 2 and roughness.shape[1] == 1:
            return roughness[:, 0]
        return roughness

    _SEGMENT_SAMPLES: int = 32

    def max_height_along_segment(self, p0_xy: Tensor, p1_xy: Tensor) -> Tensor:
        p0_xy = torch.as_tensor(p0_xy, device=self.heightmaps.device, dtype=self.heightmaps.dtype)
        p1_xy = torch.as_tensor(p1_xy, device=self.heightmaps.device, dtype=self.heightmaps.dtype)
        p0_xy, squeeze_p0 = _reshape_endpoints(p0_xy, self.batch_size)
        p1_xy, squeeze_p1 = _reshape_endpoints(p1_xy, self.batch_size)

        if p0_xy.shape[0] == 1 and self.batch_size > 1:
            p0_xy = p0_xy.expand(self.batch_size, -1)
        if p1_xy.shape[0] == 1 and self.batch_size > 1:
            p1_xy = p1_xy.expand(self.batch_size, -1)
        if p0_xy.shape[0] != self.batch_size or p1_xy.shape[0] != self.batch_size:
            raise ValueError("segment batch size must match terrain batch size")

        S = self._SEGMENT_SAMPLES
        t = torch.linspace(0.0, 1.0, S, device=p0_xy.device, dtype=p0_xy.dtype).view(1, S, 1)
        # (N, S, 2): interpolated sample points along each segment
        points = (1.0 - t) * p0_xy.unsqueeze(1) + t * p1_xy.unsqueeze(1)

        grid_xy = _normalize_points(points, self.world_x_range, self.world_y_range)
        # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
        sample_grid = grid_xy.unsqueeze(2)  # (N, S, 1, 2)
        sampled = F.grid_sample(
            self.heightmaps,
            sample_grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        heights = sampled[:, 0, :, 0]  # (N, S)

        return torch.amax(heights, dim=1)

    def batch_max_height_along_segment(self, p0_xy: Tensor, p1_xy: Tensor) -> Tensor:
        """Query max terrain height along segments for multiple legs at once.

        Args:
            p0_xy: (N, K, 2) start endpoints for K legs across N envs.
            p1_xy: (N, K, 2) end endpoints for K legs across N envs.

        Returns:
            (N, K) max heights along each segment.
        """
        p0_xy = torch.as_tensor(p0_xy, device=self.heightmaps.device, dtype=self.heightmaps.dtype)
        p1_xy = torch.as_tensor(p1_xy, device=self.heightmaps.device, dtype=self.heightmaps.dtype)

        if p0_xy.ndim != 3 or p1_xy.ndim != 3:
            raise ValueError("batch_max_height_along_segment expects (N, K, 2) inputs")
        N, K, _ = p0_xy.shape
        if N != self.batch_size:
            raise ValueError("batch size must match terrain batch size")

        S = self._SEGMENT_SAMPLES
        t = torch.linspace(0.0, 1.0, S, device=p0_xy.device, dtype=p0_xy.dtype).view(1, 1, S, 1)
        # (N, K, S, 2)
        points = (1.0 - t) * p0_xy.unsqueeze(2) + t * p1_xy.unsqueeze(2)
        # Reshape to (N, K*S, 2) for normalization and grid_sample
        points_flat = points.reshape(N, K * S, 2)

        grid_xy = _normalize_points(points_flat, self.world_x_range, self.world_y_range)
        sample_grid = grid_xy.unsqueeze(2)  # (N, K*S, 1, 2)
        sampled = F.grid_sample(
            self.heightmaps,
            sample_grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )
        heights = sampled[:, 0, :, 0]  # (N, K*S)
        heights = heights.reshape(N, K, S)

        return torch.amax(heights, dim=2)  # (N, K)


class PlannerTerrain(BatchedTerrain):
    """Formal terrain ABI for planner code.

    Use PlannerTerrain.from_ray_hits(...) to construct planner terrain instances.
    Direct constructor calls are rejected so the ABI stays explicit.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise TypeError("PlannerTerrain must be constructed with PlannerTerrain.from_ray_hits(...)")

    @classmethod
    def _from_heightmaps(
        cls,
        heightmaps: Tensor,
        *,
        world_x_range: tuple[float, float],
        world_y_range: tuple[float, float],
    ) -> "PlannerTerrain":
        self = object.__new__(cls)
        BatchedTerrain.__init__(
            self,
            heightmaps,
            world_x_range=world_x_range,
            world_y_range=world_y_range,
        )
        return self

    @classmethod
    def from_ray_hits(
        cls,
        ray_hits: Tensor,
        *,
        world_x_range: tuple[float, float] | None = None,
        world_y_range: tuple[float, float] | None = None,
    ) -> "PlannerTerrain":
        if (world_x_range is None) != (world_y_range is None):
            raise ValueError("world_x_range and world_y_range must both be provided or both omitted")
        if world_x_range is None or world_y_range is None:
            world_x_range, world_y_range = _canonical_world_ranges_from_ray_hits(ray_hits)
        else:
            _validate_ranges(world_x_range, world_y_range)

        return cls._from_heightmaps(
            _heightmaps_from_ray_hits(ray_hits),
            world_x_range=world_x_range,
            world_y_range=world_y_range,
        )


__all__ = ["PlannerTerrain", "BatchedTerrain"]
