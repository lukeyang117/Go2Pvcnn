"""Fixed single-subterrain scene for Parallelism small-obstacle RL."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ParallelismSmallObstacleSceneCfg:
    """Parameters for the one fixed small-obstacle subterrain."""

    terrain_size_m: tuple[float, float] = (8.0, 8.0)
    terrain_border_m: float = 20.0
    obstacle_patch_size_m: float = 2.0
    reset_clear_radius_m: float = 0.25
    obstacle_center_exclusion_radius_m: float = 0.30
    small_obstacle_count: int = 24
    small_obstacle_jitter_m: float = 0.03
    small_obstacle_min_spacing_m: float = 0.18
    small_obstacle_seed: int = 20260806

    def __post_init__(self) -> None:
        if self.terrain_size_m[0] <= 0.0 or self.terrain_size_m[1] <= 0.0:
            raise ValueError("terrain_size_m must be positive")
        if self.obstacle_patch_size_m <= 0.0:
            raise ValueError("obstacle_patch_size_m must be positive")
        if self.reset_clear_radius_m <= 0.0:
            raise ValueError("reset_clear_radius_m must be positive")
        if self.obstacle_center_exclusion_radius_m < self.reset_clear_radius_m:
            raise ValueError("obstacle center exclusion radius must cover reset clear radius")
        if self.small_obstacle_count <= 0:
            raise ValueError("small_obstacle_count must be positive")
        if self.small_obstacle_jitter_m < 0.0:
            raise ValueError("small_obstacle_jitter_m must be non-negative")
        if self.small_obstacle_min_spacing_m <= 0.0:
            raise ValueError("small_obstacle_min_spacing_m must be positive")


def _stable_jitter(index: int, seed: int, scale: float) -> tuple[float, float]:
    value = (index * 1103515245 + seed * 12345 + 1013904223) & 0x7FFFFFFF
    x_unit = ((value >> 4) & 0xFFFF) / 65535.0
    y_unit = ((value >> 20) & 0x7FF) / 2047.0
    return ((2.0 * x_unit - 1.0) * scale, (2.0 * y_unit - 1.0) * scale)


def build_small_obstacle_local_xy(
    cfg: ParallelismSmallObstacleSceneCfg,
) -> tuple[tuple[float, float], ...]:
    """Build one deterministic, evenly distributed square-minus-circle layout."""

    half_patch = 0.5 * cfg.obstacle_patch_size_m
    grid_step = max(0.30, cfg.small_obstacle_min_spacing_m + 0.12)
    grid_count = int(math.floor((2.0 * half_patch) / grid_step)) + 1
    grid_start = -0.5 * (grid_count - 1) * grid_step
    candidates: list[tuple[float, float, int]] = []

    for ix in range(grid_count):
        for iy in range(grid_count):
            x = grid_start + ix * grid_step
            y = grid_start + iy * grid_step
            if math.hypot(x, y) < cfg.obstacle_center_exclusion_radius_m:
                continue
            jitter_x, jitter_y = _stable_jitter(ix * grid_count + iy, cfg.small_obstacle_seed, cfg.small_obstacle_jitter_m)
            x = max(-half_patch, min(half_patch, x + jitter_x))
            y = max(-half_patch, min(half_patch, y + jitter_y))
            if math.hypot(x, y) < cfg.obstacle_center_exclusion_radius_m:
                continue
            candidates.append((x, y, ix * grid_count + iy))

    selected: list[tuple[float, float]] = []
    for x, y, _ in candidates:
        if all(math.hypot(x - px, y - py) >= cfg.small_obstacle_min_spacing_m for px, py in selected):
            selected.append((x, y))
        if len(selected) == cfg.small_obstacle_count:
            break

    if len(selected) != cfg.small_obstacle_count:
        raise ValueError(
            f"Could only place {len(selected)} of {cfg.small_obstacle_count} obstacles in the configured patch."
        )
    return tuple(selected)


def small_obstacles_terrain_cfg(cfg: ParallelismSmallObstacleSceneCfg):
    """Return exactly one flat subterrain for the fixed semantic course."""

    from isaaclab.terrains import terrain_gen

    return terrain_gen.TerrainGeneratorCfg(
        size=cfg.terrain_size_m,
        border_width=cfg.terrain_border_m,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 0.0),
        curriculum=False,
        sub_terrains={
            "small_obstacles": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        },
    )
