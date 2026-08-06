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
    small_obstacle_min_spacing_m: float = 0.12
    inner_obstacle_radius_m: float = 0.80
    inner_obstacle_ratio: float = 0.75
    inner_obstacle_min_spacing_m: float = 0.12
    outer_obstacle_min_spacing_m: float = 0.20
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
        if self.inner_obstacle_radius_m <= self.obstacle_center_exclusion_radius_m:
            raise ValueError("inner obstacle radius must be outside the center exclusion radius")
        if self.inner_obstacle_radius_m > math.sqrt(2.0) * 0.5 * self.obstacle_patch_size_m:
            raise ValueError("inner obstacle radius must fit inside the obstacle patch extent")
        if not 0.0 <= self.inner_obstacle_ratio <= 1.0:
            raise ValueError("inner_obstacle_ratio must be in [0, 1]")
        if self.inner_obstacle_min_spacing_m <= 0.0:
            raise ValueError("inner_obstacle_min_spacing_m must be positive")
        if self.outer_obstacle_min_spacing_m <= 0.0:
            raise ValueError("outer_obstacle_min_spacing_m must be positive")


def _stable_jitter(index: int, seed: int, scale: float) -> tuple[float, float]:
    value = (index * 1103515245 + seed * 12345 + 1013904223) & 0x7FFFFFFF
    x_unit = ((value >> 4) & 0xFFFF) / 65535.0
    y_unit = ((value >> 20) & 0x7FF) / 2047.0
    return ((2.0 * x_unit - 1.0) * scale, (2.0 * y_unit - 1.0) * scale)


def _halton(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / float(base)
    value = int(index)
    while value > 0:
        result += factor * (value % base)
        value //= base
        factor /= float(base)
    return result


def _zone_candidates(
    cfg: ParallelismSmallObstacleSceneCfg,
    *,
    r_min: float,
    r_max: float,
    seed_offset: int,
) -> list[tuple[float, float]]:
    half_patch = 0.5 * cfg.obstacle_patch_size_m
    candidates: list[tuple[float, float]] = []
    max_candidates = max(4096, cfg.small_obstacle_count * 256)
    for idx in range(1, max_candidates + 1):
        sample_index = idx + int(cfg.small_obstacle_seed % 997) + int(seed_offset)
        x = -half_patch + 2.0 * half_patch * _halton(sample_index, 2)
        y = -half_patch + 2.0 * half_patch * _halton(sample_index, 3)
        jitter_x, jitter_y = _stable_jitter(sample_index, cfg.small_obstacle_seed, cfg.small_obstacle_jitter_m)
        x = max(-half_patch, min(half_patch, x + jitter_x))
        y = max(-half_patch, min(half_patch, y + jitter_y))
        radius = math.hypot(x, y)
        if r_min <= radius <= r_max:
            candidates.append((x, y))
    return candidates


def _select_spaced_points(
    *,
    candidates: list[tuple[float, float]],
    target_count: int,
    selected: list[tuple[float, float]],
    spacing_m: float,
) -> None:
    for x, y in candidates:
        if all(math.hypot(x - px, y - py) >= spacing_m for px, py in selected):
            selected.append((x, y))
        if len(selected) >= target_count:
            return


def build_small_obstacle_local_xy(
    cfg: ParallelismSmallObstacleSceneCfg,
) -> tuple[tuple[float, float], ...]:
    """Build one deterministic square-minus-circle layout, denser near the reset hole."""

    half_patch = 0.5 * cfg.obstacle_patch_size_m
    inner_count = int(round(cfg.small_obstacle_count * cfg.inner_obstacle_ratio))
    inner_count = max(0, min(cfg.small_obstacle_count, inner_count))
    outer_count = cfg.small_obstacle_count - inner_count
    selected: list[tuple[float, float]] = []

    inner_candidates = _zone_candidates(
        cfg,
        r_min=cfg.obstacle_center_exclusion_radius_m,
        r_max=cfg.inner_obstacle_radius_m,
        seed_offset=0,
    )
    _select_spaced_points(
        candidates=inner_candidates,
        target_count=inner_count,
        selected=selected,
        spacing_m=cfg.inner_obstacle_min_spacing_m,
    )
    outer_candidates = _zone_candidates(
        cfg,
        r_min=cfg.inner_obstacle_radius_m,
        r_max=math.sqrt(2.0) * half_patch,
        seed_offset=4099,
    )
    _select_spaced_points(
        candidates=outer_candidates,
        target_count=cfg.small_obstacle_count,
        selected=selected,
        spacing_m=cfg.outer_obstacle_min_spacing_m,
    )
    if len(selected) != cfg.small_obstacle_count:
        raise ValueError(
            f"Could only place {len(selected)} of {cfg.small_obstacle_count} obstacles "
            f"with inner={inner_count}, outer={outer_count}. Reduce count or spacing."
        )
    return tuple(selected)


def small_obstacles_terrain_cfg(cfg: ParallelismSmallObstacleSceneCfg):
    """Return exactly one flat subterrain for the fixed semantic course."""

    import isaaclab.terrains as terrain_gen

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
