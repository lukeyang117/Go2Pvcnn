"""Terrain helpers for aligning numpy (raw) and torch (batched) planner stacks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def make_flat_terrains(
    *,
    world_x_range: tuple[float, float] = (-1.0, 1.0),
    world_y_range: tuple[float, float] = (-0.8, 0.8),
    grid_height: int = 33,
    grid_width: int = 33,
    flat_height: float = 0.0,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, torch.Tensor, tuple[float, float], tuple[float, float]]:
    """Build a flat heightmap (numpy) and matching ray hits for ``PlannerTerrain.from_ray_hits``."""
    if grid_height < 2 or grid_width < 2:
        raise ValueError("grid_height and grid_width must be at least 2 for align_corners sampling")

    x0, x1 = world_x_range
    y0, y1 = world_y_range
    xs = np.linspace(x0, x1, grid_width, dtype=np.float64)
    ys = np.linspace(y0, y1, grid_height, dtype=np.float64)

    heightmap = np.full((grid_height, grid_width), flat_height, dtype=np.float32)

    ray_hits = np.zeros((batch_size, grid_height, grid_width, 3), dtype=np.float32)
    for i in range(grid_height):
        for j in range(grid_width):
            ray_hits[:, i, j, 0] = float(xs[j])
            ray_hits[:, i, j, 1] = float(ys[i])
            ray_hits[:, i, j, 2] = float(flat_height)

    return heightmap, torch.as_tensor(ray_hits, dtype=dtype), world_x_range, world_y_range


def make_stairs_terrains(
    *,
    world_x_range: tuple[float, float] = (-1.0, 1.0),
    world_y_range: tuple[float, float] = (-0.8, 0.8),
    grid_height: int = 33,
    grid_width: int = 33,
    n_steps: int = 4,
    step_height: float = 0.05,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, torch.Tensor, tuple[float, float], tuple[float, float]]:
    """Linear ramp in +x from 0 to ``n_steps * step_height`` (simple stairs surrogate)."""
    if grid_height < 2 or grid_width < 2:
        raise ValueError("grid_height and grid_width must be at least 2")

    x0, x1 = world_x_range
    y0, y1 = world_y_range
    xs = np.linspace(x0, x1, grid_width, dtype=np.float64)
    ys = np.linspace(y0, y1, grid_height, dtype=np.float64)

    z_max = float(n_steps) * float(step_height)
    heightmap = np.zeros((grid_height, grid_width), dtype=np.float32)
    for i in range(grid_height):
        for j in range(grid_width):
            t = (float(xs[j]) - x0) / (x1 - x0) if x1 != x0 else 0.0
            heightmap[i, j] = float(t * z_max)

    ray_hits = np.zeros((batch_size, grid_height, grid_width, 3), dtype=np.float32)
    for i in range(grid_height):
        for j in range(grid_width):
            z = float(heightmap[i, j])
            ray_hits[:, i, j, 0] = float(xs[j])
            ray_hits[:, i, j, 1] = float(ys[i])
            ray_hits[:, i, j, 2] = z

    return heightmap, torch.as_tensor(ray_hits, dtype=dtype), world_x_range, world_y_range


class NumpyHeightmapTerrain:
    """Scalar-interface terrain wrapping a 2D numpy heightmap.

    Bilinear interpolation matches PyTorch ``grid_sample(align_corners=True,
    padding_mode='border')`` so raw planner queries agree with batched terrain.
    """

    def __init__(
        self,
        heightmap_np: np.ndarray,
        world_x_range: tuple[float, float],
        world_y_range: tuple[float, float],
    ) -> None:
        self._hm = np.asarray(heightmap_np, dtype=np.float64)
        if self._hm.ndim != 2:
            raise ValueError("heightmap must be 2-D")
        self._x0, self._x1 = float(world_x_range[0]), float(world_x_range[1])
        self._y0, self._y1 = float(world_y_range[0]), float(world_y_range[1])
        self._h, self._w = self._hm.shape

    def _to_pixel(self, x: float, y: float) -> tuple[float, float]:
        x = max(self._x0, min(self._x1, float(x)))
        y = max(self._y0, min(self._y1, float(y)))
        col = (x - self._x0) / (self._x1 - self._x0) * (self._w - 1) if self._x1 != self._x0 else 0.0
        row = (self._y1 - y) / (self._y1 - self._y0) * (self._h - 1) if self._y1 != self._y0 else 0.0
        return row, col

    def _bilinear(self, row_f: float, col_f: float) -> float:
        r0 = max(0, min(self._h - 2, int(math.floor(row_f))))
        c0 = max(0, min(self._w - 2, int(math.floor(col_f))))
        fr = max(0.0, min(1.0, row_f - r0))
        fc = max(0.0, min(1.0, col_f - c0))
        return float(
            self._hm[r0, c0] * (1.0 - fc) * (1.0 - fr)
            + self._hm[r0, c0 + 1] * fc * (1.0 - fr)
            + self._hm[r0 + 1, c0] * (1.0 - fc) * fr
            + self._hm[r0 + 1, c0 + 1] * fc * fr
        )

    def height_at(self, x: float, y: float) -> float:
        row, col = self._to_pixel(x, y)
        return self._bilinear(row, col)

    def roughness_at(self, x: float, y: float) -> float:
        dx = (self._x1 - self._x0) / max(self._w - 1, 1)
        dy = (self._y1 - self._y0) / max(self._h - 1, 1)
        if dx == 0.0 and dy == 0.0:
            return 0.0
        dzdx = (self.height_at(x + dx, y) - self.height_at(x - dx, y)) / (2.0 * dx) if dx > 0 else 0.0
        dzdy = (self.height_at(x, y + dy) - self.height_at(x, y - dy)) / (2.0 * dy) if dy > 0 else 0.0
        return math.sqrt(dzdx * dzdx + dzdy * dzdy)

    def max_height_along_segment(
        self,
        p0_xy: tuple[float, float],
        p1_xy: tuple[float, float],
    ) -> float:
        x0, y0 = float(p0_xy[0]), float(p0_xy[1])
        x1, y1 = float(p1_xy[0]), float(p1_xy[1])
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist == 0.0:
            return self.height_at(x0, y0)
        step = (self._x1 - self._x0) / max(self._w - 1, 1)
        n = max(3, int(math.ceil(dist / max(step, 1e-9))) * 4 + 1)
        if n % 2 == 0:
            n += 1
        max_h = float("-inf")
        for i in range(n):
            t = i / (n - 1)
            h = self.height_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if h > max_h:
                max_h = h
        return max_h


def verify_terrain_height_at_consistency(
    batched_terrain: Any,
    heightmap_2d: np.ndarray,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    atol_interior: float,
    atol_boundary: float,
) -> None:
    """Assert ``height_at`` matches discrete heightmap samples on grid-interior world points."""
    hm = np.asarray(heightmap_2d, dtype=np.float64)
    if hm.ndim != 2:
        raise ValueError("heightmap_2d must be 2D")
    h, w = hm.shape
    if h < 3 or w < 3:
        raise ValueError("heightmap_2d must be at least 3x3 to define an interior")

    x0, x1 = world_x_range
    y0, y1 = world_y_range

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            world_x = x0 + j * (x1 - x0) / (w - 1)
            world_y = y0 + i * (y1 - y0) / (h - 1)
            pts = torch.tensor([[world_x, world_y]], dtype=torch.float32)
            sampled = batched_terrain.height_at(pts)
            expected = torch.tensor([hm[i, j]], dtype=torch.float32)
            torch.testing.assert_close(sampled.cpu(), expected, atol=atol_interior, rtol=0.0)

    if atol_boundary < 0:
        raise ValueError("atol_boundary must be non-negative")

    # Boundary midpoints (excluding corners): bilinear + border padding can differ from nodal values.
    for j in range(1, w - 1):
        world_x = x0 + j * (x1 - x0) / (w - 1)
        for i, world_y in ((0, y0), (h - 1, y1)):
            pts = torch.tensor([[world_x, world_y]], dtype=torch.float32)
            sampled = batched_terrain.height_at(pts)
            expected = torch.tensor([hm[i, j]], dtype=torch.float32)
            torch.testing.assert_close(sampled.cpu(), expected, atol=atol_boundary, rtol=0.0)

    for i in range(1, h - 1):
        world_y = y0 + i * (y1 - y0) / (h - 1)
        for j, world_x in ((0, x0), (w - 1, x1)):
            pts = torch.tensor([[world_x, world_y]], dtype=torch.float32)
            sampled = batched_terrain.height_at(pts)
            expected = torch.tensor([hm[i, j]], dtype=torch.float32)
            torch.testing.assert_close(sampled.cpu(), expected, atol=atol_boundary, rtol=0.0)
