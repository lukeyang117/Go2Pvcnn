"""Semantic obstacle curriculum configuration and state helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

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
    episode_had_small_collision: torch.Tensor | None = None


def _ensure_episode_collision_state(
    state: SemanticObstacleCurriculumState,
    num_envs: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    current = state.episode_had_small_collision
    if current is None or int(current.numel()) != int(num_envs) or current.device != device:
        current = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
        state.episode_had_small_collision = current
    return current


def update_episode_small_collision_from_forces(
    state: SemanticObstacleCurriculumState,
    small_force_matrix_w: torch.Tensor,
    threshold: float,
    *,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update sticky per-episode small-obstacle collision flags from real contact forces."""

    forces = torch.as_tensor(small_force_matrix_w, dtype=torch.float32)
    if forces.ndim != 4 or int(forces.shape[-1]) != 3:
        raise ValueError(f"small_force_matrix_w must be [N,B,O,3], got {tuple(forces.shape)}")
    hit = torch.linalg.vector_norm(forces, dim=-1).gt(float(threshold)).any(dim=(1, 2))
    flags = _ensure_episode_collision_state(state, int(hit.numel()), device=hit.device)
    if env_ids is None:
        flags |= hit
        return hit
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=hit.device).reshape(-1)
    flags[ids] |= hit.index_select(0, ids)
    return hit


def update_episode_small_collision_from_map_contacts(
    state: SemanticObstacleCurriculumState,
    small_contact_mask: torch.Tensor,
    *,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update sticky per-episode small-obstacle collision flags from inferred map contacts."""

    hit = torch.as_tensor(small_contact_mask, dtype=torch.bool)
    flags = _ensure_episode_collision_state(state, int(hit.numel()), device=hit.device)
    if env_ids is None:
        flags |= hit
        return hit
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=hit.device).reshape(-1)
    flags[ids] |= hit.index_select(0, ids)
    return hit


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
    "update_episode_small_collision_from_map_contacts",
    "update_episode_small_collision_from_forces",
    "validate_semantic_obstacle_curriculum_cfg",
]
