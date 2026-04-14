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


def _heightmaps_from_ray_hits(ray_hits: Tensor) -> Tensor:
    hits = _sanitize_ray_hits(ray_hits)
    if hits.ndim == 4:
        if hits.shape[-1] != 3:
            raise ValueError("ray_hits must have shape (N, H, W, 3)")
        return hits[..., 2].unsqueeze(1).contiguous()
    if hits.ndim == 3 and hits.shape[-1] == 3:
        ray_count = int(hits.shape[1])
        side = int(round(math.sqrt(ray_count)))
        if side * side != ray_count:
            raise ValueError(f"ray count {ray_count} is not a perfect square")
        reshaped = hits.reshape(hits.shape[0], side, side, 3)
        return reshaped[..., 2].unsqueeze(1).contiguous()
    if hits.ndim == 2 and hits.shape[-1] == 3:
        ray_count = int(hits.shape[0])
        side = int(round(math.sqrt(ray_count)))
        if side * side != ray_count:
            raise ValueError(f"ray count {ray_count} is not a perfect square")
        reshaped = hits.reshape(1, side, side, 3)
        return reshaped[..., 2].unsqueeze(1).contiguous()
    raise ValueError("ray_hits must have shape (N, H, W, 3), (N, H*W, 3), or (H*W, 3)")


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

        outputs = []
        step_scale = self._step_scale()
        for idx in range(self.batch_size):
            x0 = float(p0_xy[idx, 0].item())
            y0 = float(p0_xy[idx, 1].item())
            x1 = float(p1_xy[idx, 0].item())
            y1 = float(p1_xy[idx, 1].item())
            distance = math.hypot(x1 - x0, y1 - y0)
            if distance == 0.0:
                point = torch.tensor([[x0, y0]], device=self.heightmaps.device, dtype=self.heightmaps.dtype)
                outputs.append(self._sample_map(self.heightmaps[idx : idx + 1], point)[0])
                continue

            sample_count = max(3, int(math.ceil(distance / step_scale)) * 4 + 1)
            if sample_count % 2 == 0:
                sample_count += 1

            ts = torch.linspace(0.0, 1.0, sample_count, device=self.heightmaps.device, dtype=self.heightmaps.dtype)
            xs = x0 + (x1 - x0) * ts
            ys = y0 + (y1 - y0) * ts
            pts = torch.stack((xs, ys), dim=-1).unsqueeze(0)
            heights = self._sample_map(self.heightmaps[idx : idx + 1], pts)[0]
            outputs.append(torch.max(heights))

        return torch.stack(outputs)


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
        world_x_range: tuple[float, float],
        world_y_range: tuple[float, float],
    ) -> "PlannerTerrain":
        return cls._from_heightmaps(
            _heightmaps_from_ray_hits(ray_hits),
            world_x_range=world_x_range,
            world_y_range=world_y_range,
        )


__all__ = ["PlannerTerrain", "BatchedTerrain"]
