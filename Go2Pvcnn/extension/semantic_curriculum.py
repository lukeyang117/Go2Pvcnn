"""Semantic obstacle curriculum configuration and state helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch


@dataclass(frozen=True)
class SemanticObstacleCount:
    small: int = 0
    large: int = 0


DEFAULT_PLANE_COUNTS: tuple[SemanticObstacleCount, ...] = (
    SemanticObstacleCount(small=0, large=0),
    SemanticObstacleCount(small=0, large=0),
    SemanticObstacleCount(small=1, large=0),
    SemanticObstacleCount(small=2, large=0),
    SemanticObstacleCount(small=3, large=0),
    SemanticObstacleCount(small=4, large=0),
    SemanticObstacleCount(small=5, large=1),
    SemanticObstacleCount(small=6, large=1),
    SemanticObstacleCount(small=7, large=2),
    SemanticObstacleCount(small=8, large=2),
)
DEFAULT_NON_PLANE_COUNTS: tuple[SemanticObstacleCount, ...] = (
    SemanticObstacleCount(small=0, large=0),
    SemanticObstacleCount(small=0, large=0),
    SemanticObstacleCount(small=1, large=0),
    SemanticObstacleCount(small=1, large=0),
    SemanticObstacleCount(small=2, large=0),
    SemanticObstacleCount(small=2, large=0),
    SemanticObstacleCount(small=3, large=1),
    SemanticObstacleCount(small=3, large=1),
    SemanticObstacleCount(small=4, large=1),
    SemanticObstacleCount(small=4, large=1),
)
DEFAULT_CENTER_SAFETY_HALF_EXTENT_M: tuple[float, ...] = (0.85,)
DEFAULT_MIN_SPACING_CLEARANCE_M: tuple[float, ...] = (0.15,)
DEFAULT_TILE_MARGIN_M: tuple[float, ...] = (0.50,)


@dataclass
class SemanticObstacleCurriculumCfg:
    enabled: bool = True
    plane_terrain_names: tuple[str, ...] = ("flat",)
    plane_counts: tuple[SemanticObstacleCount, ...] = field(default_factory=lambda: DEFAULT_PLANE_COUNTS)
    non_plane_counts: tuple[SemanticObstacleCount, ...] = field(default_factory=lambda: DEFAULT_NON_PLANE_COUNTS)
    center_safety_half_extent_m: tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_CENTER_SAFETY_HALF_EXTENT_M
    )
    min_spacing_clearance_m: tuple[float, ...] = field(default_factory=lambda: DEFAULT_MIN_SPACING_CLEARANCE_M)
    tile_margin_m: tuple[float, ...] = field(default_factory=lambda: DEFAULT_TILE_MARGIN_M)
    collision_force_threshold: float = 1.0
    plane_collision_rate_threshold: float = 0.03
    consecutive_success_required: int = 5

    def __post_init__(self) -> None:
        validate_semantic_obstacle_curriculum_cfg(self)


def _validate_count_sequence(values: tuple[SemanticObstacleCount, ...], *, name: str) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one row entry")
    for idx, item in enumerate(values):
        if not isinstance(item, SemanticObstacleCount):
            raise TypeError(f"{name}[{idx}] must be SemanticObstacleCount, got {type(item).__name__}")
        if int(item.small) != item.small or int(item.large) != item.large:
            raise ValueError(f"{name}[{idx}] counts must be integers, got {item!r}")
        if item.small < 0 or item.large < 0:
            raise ValueError(f"{name}[{idx}] counts must be non-negative, got {item!r}")


def _validate_float_sequence(values: tuple[float, ...], *, name: str, allowed_len: int) -> None:
    if len(values) not in (1, allowed_len):
        raise ValueError(f"{name} length must be 1 or match row count length ({allowed_len}), got {len(values)}")
    for idx, value in enumerate(values):
        value_f = float(value)
        if not math.isfinite(value_f) or value_f < 0.0:
            raise ValueError(f"{name}[{idx}] must be finite and non-negative, got {value!r}")


def validate_semantic_obstacle_curriculum_cfg(cfg: SemanticObstacleCurriculumCfg) -> None:
    _validate_count_sequence(cfg.plane_counts, name="plane_counts")
    _validate_count_sequence(cfg.non_plane_counts, name="non_plane_counts")
    row_count_len = max(len(cfg.plane_counts), len(cfg.non_plane_counts))
    _validate_float_sequence(cfg.center_safety_half_extent_m, name="center_safety_half_extent_m", allowed_len=row_count_len)
    _validate_float_sequence(cfg.min_spacing_clearance_m, name="min_spacing_clearance_m", allowed_len=row_count_len)
    _validate_float_sequence(cfg.tile_margin_m, name="tile_margin_m", allowed_len=row_count_len)
    if not all(str(name) for name in cfg.plane_terrain_names):
        raise ValueError("plane_terrain_names entries must be non-empty strings")
    if float(cfg.collision_force_threshold) < 0.0 or not math.isfinite(float(cfg.collision_force_threshold)):
        raise ValueError("collision_force_threshold must be finite and non-negative")
    rate = float(cfg.plane_collision_rate_threshold)
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError("plane_collision_rate_threshold must be in [0, 1]")
    if int(cfg.consecutive_success_required) < 1:
        raise ValueError("consecutive_success_required must be >= 1")


def clamp_row_index(row: int, count_len: int) -> int:
    if int(count_len) <= 0:
        raise ValueError(f"count_len must be positive, got {count_len}")
    return max(0, min(int(row), int(count_len) - 1))


def count_for_row(
    cfg: SemanticObstacleCurriculumCfg,
    *,
    row: int,
    terrain_name: str | None,
) -> SemanticObstacleCount:
    plane_names = {str(name) for name in cfg.plane_terrain_names}
    counts = cfg.plane_counts if terrain_name in plane_names else cfg.non_plane_counts
    return counts[clamp_row_index(row, len(counts))]


def layout_index_for_row(cfg: SemanticObstacleCurriculumCfg, row: int) -> int:
    row_count_len = max(len(cfg.plane_counts), len(cfg.non_plane_counts))
    if len(cfg.center_safety_half_extent_m) == 1:
        return 0
    return clamp_row_index(row, row_count_len)


def layout_values_for_row(cfg: SemanticObstacleCurriculumCfg, row: int) -> tuple[float, float, float]:
    center_idx = 0 if len(cfg.center_safety_half_extent_m) == 1 else layout_index_for_row(cfg, row)
    spacing_idx = 0 if len(cfg.min_spacing_clearance_m) == 1 else layout_index_for_row(cfg, row)
    margin_idx = 0 if len(cfg.tile_margin_m) == 1 else layout_index_for_row(cfg, row)
    return (
        float(cfg.center_safety_half_extent_m[center_idx]),
        float(cfg.min_spacing_clearance_m[spacing_idx]),
        float(cfg.tile_margin_m[margin_idx]),
    )


def count_to_dict(count: SemanticObstacleCount) -> dict[str, int]:
    return {"small": int(count.small), "large": int(count.large)}


@dataclass
class SemanticObstacleCurriculumState:
    consecutive_success_count: int = 0
    last_plane_collision_rate: float = 0.0

    def update_gate_from_plane_collision_rate(
        self,
        rate: float | torch.Tensor,
        cfg: SemanticObstacleCurriculumCfg,
        *,
        plane_env_count: int | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        rate_value = float(rate.detach().item()) if isinstance(rate, torch.Tensor) else float(rate)
        self.last_plane_collision_rate = rate_value
        plane_count_value = None
        if plane_env_count is not None:
            plane_count_value = (
                int(plane_env_count.detach().item()) if isinstance(plane_env_count, torch.Tensor) else int(plane_env_count)
            )

        if not bool(cfg.enabled) or plane_count_value == 0:
            return {
                "consecutive_success_count": self.consecutive_success_count,
                "plane_collision_rate": rate_value,
                "plane_env_count": 0 if plane_count_value is None else plane_count_value,
                "gate_pass": False,
                "enabled": bool(cfg.enabled),
            }

        if rate_value <= float(cfg.plane_collision_rate_threshold):
            self.consecutive_success_count += 1
        else:
            self.consecutive_success_count = 0

        return {
            "consecutive_success_count": self.consecutive_success_count,
            "plane_collision_rate": rate_value,
            "plane_env_count": 0 if plane_count_value is None else plane_count_value,
            "gate_pass": self.consecutive_success_count >= int(cfg.consecutive_success_required),
            "enabled": bool(cfg.enabled),
        }


__all__ = [
    "DEFAULT_CENTER_SAFETY_HALF_EXTENT_M",
    "DEFAULT_MIN_SPACING_CLEARANCE_M",
    "DEFAULT_NON_PLANE_COUNTS",
    "DEFAULT_PLANE_COUNTS",
    "DEFAULT_TILE_MARGIN_M",
    "SemanticObstacleCount",
    "SemanticObstacleCurriculumCfg",
    "SemanticObstacleCurriculumState",
    "clamp_row_index",
    "count_for_row",
    "count_to_dict",
    "layout_index_for_row",
    "layout_values_for_row",
    "validate_semantic_obstacle_curriculum_cfg",
]
