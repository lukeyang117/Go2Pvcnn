from __future__ import annotations

import sys
from pathlib import Path

import pytest


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
        "plane_collision_rate_threshold": 0.03,
        "consecutive_success_required": 3,
    }
    params.update(kwargs)
    return SemanticObstacleCurriculumCfg(**params)


def test_semantic_curriculum_rejects_invalid_layout_lengths() -> None:
    with pytest.raises(ValueError, match="center_safety_half_extent_m length"):
        _cfg(center_safety_half_extent_m=(0.85, 0.5))


def test_semantic_curriculum_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _cfg(plane_counts=(SemanticObstacleCount(-1, 0), SemanticObstacleCount(0, 0), SemanticObstacleCount(0, 0)))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _cfg(plane_collision_rate_threshold=1.5)

    with pytest.raises(ValueError, match="consecutive_success_required"):
        _cfg(consecutive_success_required=0)


def test_semantic_curriculum_counts_are_indexed_by_row_and_terrain_name() -> None:
    cfg = _cfg()

    assert count_for_row(cfg, row=0, terrain_name="flat") == SemanticObstacleCount(0, 0)
    assert count_for_row(cfg, row=1, terrain_name="flat") == SemanticObstacleCount(2, 0)
    assert count_for_row(cfg, row=2, terrain_name="flat") == SemanticObstacleCount(4, 1)
    assert count_for_row(cfg, row=99, terrain_name="flat") == SemanticObstacleCount(4, 1)
    assert count_for_row(cfg, row=1, terrain_name="boxes") == SemanticObstacleCount(1, 0)
    assert count_for_row(cfg, row=99, terrain_name="boxes") == SemanticObstacleCount(2, 1)


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


def test_semantic_curriculum_consecutive_success_opens_gate() -> None:
    cfg = _cfg(consecutive_success_required=3)
    state = SemanticObstacleCurriculumState()

    out0 = state.update_gate_from_plane_collision_rate(0.02, cfg, plane_env_count=8)
    assert out0["consecutive_success_count"] == 1
    assert out0["gate_pass"] is False

    state.update_gate_from_plane_collision_rate(0.01, cfg, plane_env_count=8)
    out2 = state.update_gate_from_plane_collision_rate(0.0, cfg, plane_env_count=8)
    assert out2["consecutive_success_count"] == 3
    assert out2["gate_pass"] is True


def test_semantic_curriculum_failure_resets_success_count() -> None:
    cfg = _cfg()
    state = SemanticObstacleCurriculumState()

    state.update_gate_from_plane_collision_rate(0.02, cfg, plane_env_count=8)
    out = state.update_gate_from_plane_collision_rate(0.20, cfg, plane_env_count=8)

    assert out["consecutive_success_count"] == 0
    assert out["gate_pass"] is False


def test_semantic_curriculum_disabled_does_not_upgrade() -> None:
    cfg = _cfg(enabled=False, consecutive_success_required=1)
    state = SemanticObstacleCurriculumState()

    out = state.update_gate_from_plane_collision_rate(0.0, cfg, plane_env_count=8)

    assert out["enabled"] is False
    assert out["consecutive_success_count"] == 0
    assert out["gate_pass"] is False


def test_semantic_count_to_dict() -> None:
    assert count_to_dict(SemanticObstacleCount(small=4, large=1)) == {"small": 4, "large": 1}
