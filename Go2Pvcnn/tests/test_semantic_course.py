from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.semantic_course import (
    DEFAULT_VIEWER_REPRESENTATIVE_STAGE,
    LARGE_OBSTACLE_SIZE,
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    SMALL_OBSTACLE_SIZE,
    SemanticCourseStage,
    build_course_anchors,
    course_anchor_counts,
    ground_course_anchors,
    representative_rows,
    set_scene_env_to_representative_stage,
    stage_for_row,
)


def _terrain_origins(num_rows: int = 10, num_cols: int = 2):
    origins = []
    for row in range(num_rows):
        origins.append([])
        for col in range(num_cols):
            origins[row].append((row * 8.0, col * 8.0, row * 0.1 + col * 0.01))
    return origins


def test_row_mapping_and_representative_rows_are_deterministic():
    assert stage_for_row(0, 10) is SemanticCourseStage.S1
    assert stage_for_row(2, 10) is SemanticCourseStage.S1
    assert stage_for_row(3, 10) is SemanticCourseStage.S2
    assert stage_for_row(4, 10) is SemanticCourseStage.S2
    assert stage_for_row(5, 10) is SemanticCourseStage.S3
    assert stage_for_row(7, 10) is SemanticCourseStage.S3
    assert stage_for_row(8, 10) is SemanticCourseStage.S4
    assert stage_for_row(9, 10) is SemanticCourseStage.S4

    assert representative_rows(10) == {
        SemanticCourseStage.S1: 1,
        SemanticCourseStage.S2: 3,
        SemanticCourseStage.S3: 6,
        SemanticCourseStage.S4: 8,
    }
    assert DEFAULT_VIEWER_REPRESENTATIVE_STAGE is SemanticCourseStage.S4


@pytest.mark.parametrize(
    ("stage", "small_count", "large_count"),
    (
        (SemanticCourseStage.S1, 0, 0),
        (SemanticCourseStage.S2, 4, 0),
        (SemanticCourseStage.S3, 4, 1),
        (SemanticCourseStage.S4, 6, 1),
    ),
)
def test_stage_layout_counts_are_exact(stage, small_count, large_count):
    counts = course_anchor_counts(stage)
    assert counts == {"small": small_count, "large": large_count}


def test_build_course_anchors_uses_stable_roots_and_exact_counts():
    terrain_origins = _terrain_origins(num_rows=10, num_cols=1)
    anchors = build_course_anchors(terrain_origins)

    s2_small = [a for a in anchors if a.row == 3 and a.semantic_class == "small"]
    s3_small = [a for a in anchors if a.row == 5 and a.semantic_class == "small"]
    s3_large = [a for a in anchors if a.row == 5 and a.semantic_class == "large"]
    s4_small = [a for a in anchors if a.row == 8 and a.semantic_class == "small"]
    s4_large = [a for a in anchors if a.row == 8 and a.semantic_class == "large"]

    assert len([a for a in anchors if a.row == 1]) == 0
    assert len(s2_small) == 4
    assert len(s3_small) == 4
    assert len(s3_large) == 1
    assert len(s4_small) == 6
    assert len(s4_large) == 1
    assert all(anchor.prim_path.startswith(SEMANTIC_COURSE_SMALL_ROOT) for anchor in s2_small + s3_small + s4_small)
    assert all(anchor.prim_path.startswith(SEMANTIC_COURSE_LARGE_ROOT) for anchor in s3_large + s4_large)


def test_ground_course_anchors_uses_surface_height_plus_half_obstacle_height():
    terrain_origins = _terrain_origins(num_rows=10, num_cols=1)
    anchors = build_course_anchors(terrain_origins)
    one_small = next(anchor for anchor in anchors if anchor.semantic_class == "small")
    one_large = next(anchor for anchor in anchors if anchor.semantic_class == "large")

    grounded = ground_course_anchors(
        [one_small, one_large],
        terrain_height_at_xy=lambda x, y: 0.25 + 0.01 * x - 0.02 * y,
    )

    small_obstacle, large_obstacle = grounded
    expected_small_z = 0.25 + 0.01 * one_small.world_xy[0] - 0.02 * one_small.world_xy[1] + SMALL_OBSTACLE_SIZE[2] * 0.5
    expected_large_z = 0.25 + 0.01 * one_large.world_xy[0] - 0.02 * one_large.world_xy[1] + LARGE_OBSTACLE_SIZE[2] * 0.5

    assert small_obstacle.world_center[:2] == one_small.world_xy
    assert large_obstacle.world_center[:2] == one_large.world_xy
    assert small_obstacle.world_center[2] == pytest.approx(expected_small_z)
    assert large_obstacle.world_center[2] == pytest.approx(expected_large_z)


def test_set_scene_env_to_representative_stage_moves_env_origin_to_target_row():
    class FakeTerrain:
        def __init__(self):
            self.terrain_origins = torch.tensor(_terrain_origins(num_rows=10, num_cols=2), dtype=torch.float32)
            self.terrain_types = torch.tensor([1, 0], dtype=torch.long)
            self.terrain_levels = torch.zeros(2, dtype=torch.long)
            self.env_origins = torch.zeros(2, 3, dtype=torch.float32)

    class FakeScene:
        def __init__(self):
            self.terrain = FakeTerrain()

    scene = FakeScene()
    row = set_scene_env_to_representative_stage(scene, env_id=0, stage=SemanticCourseStage.S3)

    assert row == 6
    torch.testing.assert_close(scene.terrain.env_origins[0], scene.terrain.terrain_origins[6, 1])
    assert int(scene.terrain.terrain_levels[0].item()) == 6
