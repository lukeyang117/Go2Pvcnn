from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


def _install_fake_isaaclab(monkeypatch) -> None:
    isaaclab_module = types.ModuleType("isaaclab")
    terrains_module = types.ModuleType("isaaclab.terrains")

    class TerrainImporter:
        pass

    terrains_module.TerrainImporter = TerrainImporter
    isaaclab_module.terrains = terrains_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.terrains", terrains_module)


def test_semantic_course_curriculum_fallback_stage_counts(monkeypatch) -> None:
    _install_fake_isaaclab(monkeypatch)
    from extension.semantic_course import SemanticCourseStage, semantic_counts_for_tile

    assert semantic_counts_for_tile(
        row=9,
        col=0,
        terrain_names=("flat",),
        curriculum_cfg=None,
        fallback_stage=SemanticCourseStage.S4,
    ) == {"small": 6, "large": 1}


def test_semantic_course_curriculum_uses_plane_and_non_plane_counts(monkeypatch) -> None:
    _install_fake_isaaclab(monkeypatch)
    from extension.semantic_course import SemanticCourseStage, semantic_counts_for_tile
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(5, 2)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(3, 1)),
        center_safety_half_extent_m=(0.8, 0.2),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.3),
    )

    assert semantic_counts_for_tile(
        row=1,
        col=0,
        terrain_names=("flat", "boxes"),
        curriculum_cfg=cfg,
        fallback_stage=SemanticCourseStage.S1,
    ) == {"small": 5, "large": 2}
    assert semantic_counts_for_tile(
        row=1,
        col=1,
        terrain_names=("flat", "boxes"),
        curriculum_cfg=cfg,
        fallback_stage=SemanticCourseStage.S1,
    ) == {"small": 3, "large": 1}
    assert semantic_counts_for_tile(
        row=99,
        col=0,
        terrain_names=("flat", "boxes"),
        curriculum_cfg=cfg,
        fallback_stage=SemanticCourseStage.S1,
    ) == {"small": 5, "large": 2}


def test_semantic_course_repeats_single_terrain_name_across_all_columns(monkeypatch) -> None:
    _install_fake_isaaclab(monkeypatch)
    from extension.semantic_course import SemanticCourseStage, semantic_counts_for_tile
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(8, 0),),
        non_plane_counts=(SemanticObstacleCount(0, 0),),
        center_safety_half_extent_m=(0.8,),
        min_spacing_clearance_m=(0.2,),
        tile_margin_m=(0.5,),
        plane_terrain_names=("flat",),
    )

    for col in (0, 1, 19):
        assert semantic_counts_for_tile(
            row=0,
            col=col,
            terrain_names=("flat",),
            curriculum_cfg=cfg,
            fallback_stage=SemanticCourseStage.S1,
        ) == {"small": 8, "large": 0}


def test_semantic_course_uses_generated_column_proportions_for_obstacle_counts(monkeypatch) -> None:
    _install_fake_isaaclab(monkeypatch)
    from extension.semantic_course import SemanticCourseStage, semantic_counts_for_tile
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    cfg = SemanticObstacleCurriculumCfg(
        plane_terrain_names=("flat", "flat_dense_small_obstacles"),
        plane_counts=(SemanticObstacleCount(0, 0),),
        non_plane_counts=(SemanticObstacleCount(5, 2),),
        terrain_obstacle_count_overrides={
            "flat": SemanticObstacleCount(0, 2),
            "flat_dense_small_obstacles": SemanticObstacleCount(40, 0),
        },
    )
    generator = SimpleNamespace(
        num_cols=20,
        sub_terrains={
            "flat_dense_small_obstacles": SimpleNamespace(proportion=0.05),
            "flat": SimpleNamespace(proportion=0.05),
            "random_rough": SimpleNamespace(proportion=0.1),
            "hf_pyramid_slope": SimpleNamespace(proportion=0.1),
            "hf_pyramid_slope_inv": SimpleNamespace(proportion=0.1),
            "boxes": SimpleNamespace(proportion=0.2),
            "pyramid_stairs": SimpleNamespace(proportion=0.2),
            "pyramid_stairs_inv": SimpleNamespace(proportion=0.2),
        },
    )

    assert semantic_counts_for_tile(
        row=0,
        col=0,
        terrain_names=tuple(generator.sub_terrains),
        curriculum_cfg=cfg,
        fallback_stage=SemanticCourseStage.S1,
        terrain_generator=generator,
    ) == {"small": 40, "large": 0}
    assert semantic_counts_for_tile(
        row=0,
        col=1,
        terrain_names=tuple(generator.sub_terrains),
        curriculum_cfg=cfg,
        fallback_stage=SemanticCourseStage.S1,
        terrain_generator=generator,
    ) == {"small": 0, "large": 2}


def test_semantic_course_curriculum_layout_row(monkeypatch) -> None:
    _install_fake_isaaclab(monkeypatch)
    from extension.semantic_course import SemanticCourseLayoutCfg, layout_cfg_for_row
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(5, 2)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(3, 1)),
        center_safety_half_extent_m=(0.8, 0.2),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.3),
    )
    base = SemanticCourseLayoutCfg(max_layout_attempts=17)

    out = layout_cfg_for_row(base, cfg, 99)

    assert out.center_safety_half_extent_m == pytest.approx(0.2)
    assert out.min_spacing_clearance_m == pytest.approx(0.1)
    assert out.tile_margin_m == pytest.approx(0.3)
    assert out.max_layout_attempts == 17


def test_build_course_anchors_curriculum_counts_and_paths(monkeypatch) -> None:
    _install_fake_isaaclab(monkeypatch)
    from extension.semantic_course import build_course_anchors
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(2, 1)),
        non_plane_counts=(SemanticObstacleCount(0, 0), SemanticObstacleCount(1, 0)),
        center_safety_half_extent_m=(0.8, 0.2),
        min_spacing_clearance_m=(0.2, 0.1),
        tile_margin_m=(0.5, 0.3),
    )
    terrain_origins = [
        [(0.0, 0.0, 0.0), (0.0, 8.0, 0.0)],
        [(8.0, 0.0, 0.0), (8.0, 8.0, 0.0)],
    ]

    anchors = build_course_anchors(
        terrain_origins,
        tile_size=(8.0, 8.0),
        terrain_names=("flat", "boxes"),
        semantic_curriculum_cfg=cfg,
    )

    flat = [a for a in anchors if a.col == 0]
    non_flat = [a for a in anchors if a.col == 1]
    assert sum(a.semantic_class == "small" for a in flat) == 2
    assert sum(a.semantic_class == "large" for a in flat) == 1
    assert sum(a.semantic_class == "small" for a in non_flat) == 1
    assert sum(a.semantic_class == "large" for a in non_flat) == 0
    assert all("/row_" in a.prim_path and "/col_" in a.prim_path and "/slot_" in a.prim_path for a in anchors)


def test_build_course_anchors_single_flat_subterrain_populates_every_column(monkeypatch) -> None:
    _install_fake_isaaclab(monkeypatch)
    from extension.semantic_course import build_course_anchors
    from extension.semantic_curriculum import SemanticObstacleCount, SemanticObstacleCurriculumCfg

    cfg = SemanticObstacleCurriculumCfg(
        plane_counts=(SemanticObstacleCount(2, 0),),
        non_plane_counts=(SemanticObstacleCount(0, 0),),
        center_safety_half_extent_m=(0.8,),
        min_spacing_clearance_m=(0.2,),
        tile_margin_m=(0.5,),
        plane_terrain_names=("flat",),
    )
    terrain_origins = [
        [(float(row) * 8.0, float(col) * 8.0, 0.0) for col in range(20)]
        for row in range(10)
    ]

    anchors = build_course_anchors(
        terrain_origins,
        tile_size=(8.0, 8.0),
        terrain_names=("flat",),
        semantic_curriculum_cfg=cfg,
    )

    small_counts_by_col = {
        col: sum(1 for anchor in anchors if anchor.row == 0 and anchor.col == col and anchor.semantic_class == "small")
        for col in range(20)
    }
    assert small_counts_by_col == {col: 2 for col in range(20)}
