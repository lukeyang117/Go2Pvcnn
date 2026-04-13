"""Import-safe heightmap adapter scaffolding for planner integration.

This module stays free of Isaac Lab imports and provides small helpers for
normalizing and inspecting pure-Python heightmap grids.

`LocalGridTerrain` implements the surface API expected by
``raw/kinematic_footsteps/scripts/go2fp`` (``height_at``, ``roughness_at``,
``max_height_along_segment``) from a regular height field in robot-local
horizontal coordinates (+x forward, +y left) anchored at a world-space origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

GridRow = Sequence[float]
HeightmapGrid = Sequence[GridRow]


def is_rectangular_heightmap(grid: HeightmapGrid) -> bool:
    """Return True when every row has the same length."""
    if not grid:
        return True
    width = len(grid[0])
    return all(len(row) == width for row in grid)


def heightmap_shape(grid: HeightmapGrid) -> tuple[int, int]:
    """Return heightmap rows and columns."""
    if not grid:
        return (0, 0)
    if not is_rectangular_heightmap(grid):
        raise ValueError("heightmap grid must be rectangular")
    return (len(grid), len(grid[0]))


@dataclass(frozen=True, slots=True)
class HeightmapAdapterConfig:
    """Configuration scaffold for future Isaac heightmap extraction."""

    resolution_m: float = 0.05
    size_m: float = 5.0
    origin_xy: tuple[float, float] = (0.0, 0.0)
    invalid_value: float = 0.0

    def cell_count(self) -> int:
        """Return the nominal number of cells per side."""
        if self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")
        if self.size_m <= 0.0:
            raise ValueError("size_m must be positive")
        return max(1, int(round(self.size_m / self.resolution_m)))


def _extract_yaw_wxyz(quat_wxyz: np.ndarray) -> float:
    """Yaw (about +Z) from a single wxyz quaternion."""
    w, x, y, z = (float(quat_wxyz[i]) for i in range(4))
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _world_xy_to_local(wx: float, wy: float, origin_xy: tuple[float, float], yaw: float) -> tuple[float, float]:
    dx = float(wx) - float(origin_xy[0])
    dy = float(wy) - float(origin_xy[1])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    return lx, ly


class LocalGridTerrain:
    """Height field on a robot-local rectilinear grid, exposed to raw go2fp."""

    __slots__ = ("_z", "_sx", "_sy", "_nx", "_ny", "_origin_xy", "_yaw", "_fill")

    def __init__(
        self,
        z_grid_local: np.ndarray,
        size_xy: tuple[float, float],
        origin_xy_world: tuple[float, float],
        yaw_world: float,
        *,
        fill_value: float = 0.0,
    ) -> None:
        z = np.asarray(z_grid_local, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"z_grid_local must be 2D, got {z.ndim}")
        self._ny, self._nx = int(z.shape[0]), int(z.shape[1])
        if self._nx < 1 or self._ny < 1:
            raise ValueError("z_grid_local must be non-empty")
        self._z = np.where(np.isfinite(z), z, float(fill_value)).astype(np.float64)
        self._sx = float(size_xy[0])
        self._sy = float(size_xy[1])
        self._origin_xy = (float(origin_xy_world[0]), float(origin_xy_world[1]))
        self._yaw = float(yaw_world)
        self._fill = float(fill_value)

    @classmethod
    def from_world_ray_hits(
        cls,
        hits_w: np.ndarray,
        *,
        root_pos_w: np.ndarray,
        root_quat_w: np.ndarray,
        size_xy: tuple[float, float],
        fill_value: float = 0.0,
    ) -> "LocalGridTerrain":
        """Bin ray hit positions into a local regular grid (handles arbitrary ray order).

        Args:
            hits_w: Array of shape ``(side, side, 3)`` or ``(n_rays, 3)`` with world XYZ.
            root_pos_w: Shape ``(3,)`` base position in world.
            root_quat_w: Shape ``(4,)`` wxyz root orientation.
            size_xy: Physical span (meters) along local x and y covered by the scanner pattern.
        """
        hits = np.asarray(hits_w, dtype=np.float64)
        if hits.ndim == 3:
            ny_in, nx_in, three = hits.shape
            if three != 3:
                raise ValueError("hits_w last dim must be 3")
            flat = hits.reshape(-1, 3)
            side = int(round(math.sqrt(flat.shape[0])))
            if side * side != flat.shape[0]:
                raise ValueError(f"ray count {flat.shape[0]} is not a perfect square")
            if ny_in != side or nx_in != side:
                raise ValueError(f"hits_w shape {ny_in}x{nx_in} inconsistent with square side {side}")
            nx = ny = side
        elif hits.ndim == 2 and hits.shape[-1] == 3:
            flat = hits.reshape(-1, 3)
            side = int(round(math.sqrt(flat.shape[0])))
            if side * side != flat.shape[0]:
                raise ValueError(f"ray count {flat.shape[0]} is not a perfect square")
            nx = ny = side
        else:
            raise ValueError(f"hits_w must be (side,side,3) or (n,3), got {hits.shape}")

        root = np.asarray(root_pos_w, dtype=np.float64).reshape(3)
        quat = np.asarray(root_quat_w, dtype=np.float64).reshape(4)
        yaw = _extract_yaw_wxyz(quat)
        origin_xy = (float(root[0]), float(root[1]))
        sx, sy = float(size_xy[0]), float(size_xy[1])

        z_acc = np.zeros((ny, nx), dtype=np.float64)
        w_acc = np.zeros((ny, nx), dtype=np.float64)

        for p in flat:
            if not np.all(np.isfinite(p)):
                continue
            lx, ly = _world_xy_to_local(float(p[0]), float(p[1]), origin_xy, yaw)
            if nx == 1:
                ix = 0
            else:
                ix = int(round((lx + 0.5 * sx) / sx * (nx - 1)))
            if ny == 1:
                iy = 0
            else:
                iy = int(round((ly + 0.5 * sy) / sy * (ny - 1)))
            ix = int(np.clip(ix, 0, nx - 1))
            iy = int(np.clip(iy, 0, ny - 1))
            z_acc[iy, ix] += float(p[2])
            w_acc[iy, ix] += 1.0

        mask = w_acc > 1e-9
        z_grid = np.full((ny, nx), fill_value, dtype=np.float64)
        z_grid[mask] = z_acc[mask] / w_acc[mask]
        if not np.any(mask) and flat.shape[0] > 0:
            zvals = flat[:, 2]
            zvals = zvals[np.isfinite(zvals)]
            fill_z = float(np.mean(zvals)) if zvals.size > 0 else float(fill_value)
            z_grid[:, :] = fill_z
        return cls(z_grid, size_xy, origin_xy, yaw, fill_value=fill_value)

    def _local_to_grid_uv(self, lx: float, ly: float) -> tuple[float, float]:
        """Continuous grid coordinates (u along x index, v along y index)."""
        if self._nx == 1:
            u = 0.0
        else:
            u = (lx + 0.5 * self._sx) / self._sx * (self._nx - 1)
        if self._ny == 1:
            v = 0.0
        else:
            v = (ly + 0.5 * self._sy) / self._sy * (self._ny - 1)
        return u, v

    def _bilinear_z(self, u: float, v: float) -> float:
        u = float(np.clip(u, 0.0, self._nx - 1))
        v = float(np.clip(v, 0.0, self._ny - 1))
        u0 = int(math.floor(u))
        v0 = int(math.floor(v))
        u1 = min(u0 + 1, self._nx - 1)
        v1 = min(v0 + 1, self._ny - 1)
        tu = u - u0
        tv = v - v0
        z00 = self._z[v0, u0]
        z10 = self._z[v0, u1]
        z01 = self._z[v1, u0]
        z11 = self._z[v1, u1]
        top = (1.0 - tu) * z00 + tu * z10
        bottom = (1.0 - tu) * z01 + tu * z11
        return float((1.0 - tv) * top + tv * bottom)

    def height_at(self, x: float, y: float) -> float:
        lx, ly = _world_xy_to_local(float(x), float(y), self._origin_xy, self._yaw)
        u, v = self._local_to_grid_uv(lx, ly)
        return self._bilinear_z(u, v)

    def roughness_at(self, x: float, y: float) -> float:
        """Magnitude of height gradient via small world-space steps along local axes."""
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        eps = 5e-3
        dzx = (
            self.height_at(float(x) + c * eps, float(y) + s * eps)
            - self.height_at(float(x) - c * eps, float(y) - s * eps)
        ) / (2.0 * eps)
        dzy = (
            self.height_at(float(x) - s * eps, float(y) + c * eps)
            - self.height_at(float(x) + s * eps, float(y) - c * eps)
        ) / (2.0 * eps)
        return float(math.sqrt(dzx * dzx + dzy * dzy))

    def max_height_along_segment(
        self,
        p0_xy: tuple[float, float],
        p1_xy: tuple[float, float],
    ) -> float:
        samples = 12
        x0, y0 = float(p0_xy[0]), float(p0_xy[1])
        x1, y1 = float(p1_xy[0]), float(p1_xy[1])
        best = max(self.height_at(x0, y0), self.height_at(x1, y1))
        for k in range(1, samples):
            t = k / samples
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            best = max(best, self.height_at(x, y))
        return float(best)


