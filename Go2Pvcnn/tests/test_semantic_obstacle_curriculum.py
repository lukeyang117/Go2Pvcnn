from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.semantic_curriculum import (
    SemanticObstacleCount,
    SemanticObstacleCurriculumCfg,
    SemanticObstacleCurriculumState,
    count_for_row,
    count_to_dict,
    update_episode_small_collision_from_forces,
    layout_index_for_row,
    layout_values_for_row,
)


def _cfg(**kwargs) -> SemanticObstacleCurriculumCfg:
    params = {
        "plane_counts": (
            SemanticObstacleCount(0, 0),
            SemanticObstacleCount(2, 0),
            SemanticObstacleCount(4, 1),
        ),
        "non_plane_counts": (
            SemanticObstacleCount(0, 0),
            SemanticObstacleCount(1, 0),
            SemanticObstacleCount(2, 1),
        ),
        "center_safety_half_extent_m": (0.85, 0.5, 0.25),
        "min_spacing_clearance_m": (0.25, 0.18, 0.10),
        "tile_margin_m": (0.50, 0.40, 0.30),
    }
    params.update(kwargs)
    return SemanticObstacleCurriculumCfg(**params)


def test_semantic_curriculum_rejects_invalid_layout_lengths() -> None:
    with pytest.raises(ValueError, match="center_safety_half_extent_m length"):
        _cfg(center_safety_half_extent_m=(0.85, 0.5))


def test_semantic_curriculum_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _cfg(plane_counts=(SemanticObstacleCount(-1, 0), SemanticObstacleCount(0, 0), SemanticObstacleCount(0, 0)))

    with pytest.raises(ValueError, match="collision_force_threshold"):
        _cfg(collision_force_threshold=-1.0)


def test_semantic_curriculum_counts_are_indexed_by_row_and_terrain_name() -> None:
    cfg = _cfg()

    assert count_for_row(cfg, row=0, terrain_name="flat") == SemanticObstacleCount(0, 0)
    assert count_for_row(cfg, row=1, terrain_name="flat") == SemanticObstacleCount(2, 0)
    assert count_for_row(cfg, row=2, terrain_name="flat") == SemanticObstacleCount(4, 1)
    assert count_for_row(cfg, row=99, terrain_name="flat") == SemanticObstacleCount(4, 1)
    assert count_for_row(cfg, row=1, terrain_name="boxes") == SemanticObstacleCount(1, 0)
    assert count_for_row(cfg, row=99, terrain_name="boxes") == SemanticObstacleCount(2, 1)


def test_semantic_curriculum_named_terrain_override_wins_over_row_counts() -> None:
    cfg = _cfg(
        plane_counts=(SemanticObstacleCount(5, 2),),
        non_plane_counts=(SemanticObstacleCount(5, 2),),
        terrain_obstacle_count_overrides={
            "flat_dense_small_obstacles": SemanticObstacleCount(40, 0),
        },
        center_safety_half_extent_m=(0.25,),
        min_spacing_clearance_m=(0.08,),
        tile_margin_m=(0.50,),
    )

    assert count_for_row(cfg, row=0, terrain_name="flat_dense_small_obstacles") == SemanticObstacleCount(40, 0)
    assert count_for_row(cfg, row=0, terrain_name="boxes") == SemanticObstacleCount(5, 2)


def test_semantic_curriculum_layout_values_can_be_scalar_or_row_indexed() -> None:
    scalar_cfg = _cfg(
        center_safety_half_extent_m=(0.7,),
        min_spacing_clearance_m=(0.2,),
        tile_margin_m=(0.4,),
    )
    assert layout_index_for_row(scalar_cfg, 99) == 0
    assert layout_values_for_row(scalar_cfg, 99) == pytest.approx((0.7, 0.2, 0.4))

    row_cfg = _cfg()
    assert layout_index_for_row(row_cfg, 99) == 2
    assert layout_values_for_row(row_cfg, 99) == pytest.approx((0.25, 0.10, 0.30))


def test_episode_small_collision_stays_true_until_reset() -> None:
    state = SemanticObstacleCurriculumState()
    small_force = torch.zeros((3, 2, 4, 3), dtype=torch.float32)
    small_force[1, 0, 0, 0] = 2.0

    hit = update_episode_small_collision_from_forces(state, small_force, threshold=1.0)

    assert hit.tolist() == [False, True, False]
    assert state.episode_had_small_collision.tolist() == [False, True, False]

    update_episode_small_collision_from_forces(state, torch.zeros_like(small_force), threshold=1.0)

    assert state.episode_had_small_collision.tolist() == [False, True, False]


def test_semantic_count_to_dict() -> None:
    assert count_to_dict(SemanticObstacleCount(small=4, large=1)) == {"small": 4, "large": 1}
