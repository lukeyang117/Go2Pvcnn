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
    SHAPE_AXIS_Z,
    SHARED_NATIVE_SHAPE_POOL,
    SMALL_OBSTACLE_SIZE,
    SemanticCourseStage,
    bottom_to_center_offset,
    build_course_anchors,
    course_anchor_counts,
    deterministic_shape_key,
    ground_course_anchors,
    representative_rows,
    select_shape_kind,
    shape_params_for_profile,
    semantic_scale_profile,
    set_scene_env_to_representative_stage,
    stage_for_row,
    _shape_spawn_cfg,
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
    assert all(anchor.shape_kind in SHARED_NATIVE_SHAPE_POOL for anchor in anchors)


def test_deterministic_shape_selector_is_stable_and_shared_across_classes():
    key_a = deterministic_shape_key(
        stage=SemanticCourseStage.S4,
        row=8,
        col=1,
        slot_index=3,
        semantic_class="small",
    )
    key_b = deterministic_shape_key(
        stage=SemanticCourseStage.S4,
        row=8,
        col=1,
        slot_index=3,
        semantic_class="small",
    )
    key_large = deterministic_shape_key(
        stage=SemanticCourseStage.S4,
        row=8,
        col=1,
        slot_index=3,
        semantic_class="large",
    )

    assert key_a == key_b
    assert key_a != key_large
    assert select_shape_kind(
        stage=SemanticCourseStage.S4,
        row=8,
        col=1,
        slot_index=3,
        semantic_class="small",
    ) == select_shape_kind(
        stage=SemanticCourseStage.S4,
        row=8,
        col=1,
        slot_index=3,
        semantic_class="small",
    )
    assert {
        select_shape_kind(
            stage=SemanticCourseStage.S4,
            row=8,
            col=1,
            slot_index=slot_index,
            semantic_class="small",
        )
        for slot_index in range(len(SHARED_NATIVE_SHAPE_POOL))
    } == set(SHARED_NATIVE_SHAPE_POOL)


@pytest.mark.parametrize(
    ("semantic_class", "expected"),
    (
        ("small", (0.12, 0.22)),
        ("large", (0.45, 0.55)),
    ),
)
def test_semantic_scale_profile_matches_approved_sizes(semantic_class, expected):
    assert semantic_scale_profile(semantic_class) == expected


@pytest.mark.parametrize(
    ("shape_kind", "target_diameter", "target_height", "expected"),
    (
        ("sphere", 0.12, 0.22, {"radius": 0.06}),
        ("cuboid", 0.45, 0.55, {"size": (0.45, 0.45, 0.55)}),
        ("cylinder", 0.12, 0.22, {"radius": 0.06, "height": 0.22, "axis": SHAPE_AXIS_Z}),
        ("capsule", 0.12, 0.22, {"radius": 0.06, "height": 0.10, "axis": SHAPE_AXIS_Z}),
        ("cone", 0.45, 0.55, {"radius": 0.225, "height": 0.55, "axis": SHAPE_AXIS_Z}),
    ),
)
def test_shape_params_follow_native_mapping(shape_kind, target_diameter, target_height, expected):
    assert shape_params_for_profile(
        shape_kind,
        target_diameter=target_diameter,
        target_height=target_height,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("shape_kind", "shape_params", "expected_offset"),
    (
        ("sphere", {"radius": 0.06}, 0.06),
        ("cuboid", {"size": (0.12, 0.12, 0.22)}, 0.11),
        ("cylinder", {"radius": 0.06, "height": 0.22, "axis": SHAPE_AXIS_Z}, 0.11),
        ("capsule", {"radius": 0.06, "height": 0.10, "axis": SHAPE_AXIS_Z}, 0.11),
        ("cone", {"radius": 0.225, "height": 0.55, "axis": SHAPE_AXIS_Z}, 0.275),
    ),
)
def test_grounding_offsets_are_shape_aware(shape_kind, shape_params, expected_offset):
    assert bottom_to_center_offset(shape_kind, shape_params) == pytest.approx(expected_offset)


def test_ground_course_anchors_uses_surface_height_plus_shape_aware_offset():
    terrain_origins = _terrain_origins(num_rows=10, num_cols=1)
    anchors = build_course_anchors(terrain_origins)
    shape_kinds = ("sphere", "cuboid", "cylinder", "capsule", "cone")
    selected = [next(anchor for anchor in anchors if anchor.shape_kind == shape_kind) for shape_kind in shape_kinds]

    grounded = ground_course_anchors(
        selected,
        terrain_height_at_xy=lambda x, y: 0.25 + 0.01 * x - 0.02 * y,
    )

    for anchor, obstacle in zip(selected, grounded, strict=True):
        expected_z = 0.25 + 0.01 * anchor.world_xy[0] - 0.02 * anchor.world_xy[1] + anchor.ground_offset
        assert obstacle.world_center[:2] == anchor.world_xy
        assert obstacle.world_center[2] == pytest.approx(expected_z)
        assert obstacle.shape_kind == anchor.shape_kind
        assert obstacle.shape_params == anchor.shape_params


def test_spawn_cfg_dispatch_builds_matching_native_shape_cfgs():
    class _Cfg:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.func = lambda *args, **kwargs: None

    class FakeSimUtils:
        SphereCfg = _Cfg
        CuboidCfg = _Cfg
        CylinderCfg = _Cfg
        CapsuleCfg = _Cfg
        ConeCfg = _Cfg

        class RigidBodyPropertiesCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class MassPropertiesCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class CollisionPropertiesCfg:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    terrain_origins = _terrain_origins(num_rows=10, num_cols=1)
    anchors = build_course_anchors(terrain_origins)

    for shape_kind in SHARED_NATIVE_SHAPE_POOL:
        anchor = next(anchor for anchor in anchors if anchor.shape_kind == shape_kind)
        obstacle = ground_course_anchors([anchor], terrain_height_at_xy=lambda *_: 0.0)[0]
        cfg = _shape_spawn_cfg(obstacle, sim_utils=FakeSimUtils)
        assert cfg.kwargs["rigid_props"].kwargs == {"kinematic_enabled": True, "disable_gravity": True}
        assert cfg.kwargs["mass_props"].kwargs == {"mass": 1.0}
        assert cfg.kwargs["collision_props"].kwargs == {}
        if shape_kind == "sphere":
            assert cfg.kwargs["radius"] == pytest.approx(anchor.target_diameter / 2.0)
        elif shape_kind == "cuboid":
            assert cfg.kwargs["size"] == pytest.approx(
                (anchor.target_diameter, anchor.target_diameter, anchor.target_height)
            )
        else:
            assert cfg.kwargs["radius"] == pytest.approx(anchor.target_diameter / 2.0)
            assert cfg.kwargs["axis"] == SHAPE_AXIS_Z
            if shape_kind == "capsule":
                assert cfg.kwargs["height"] == pytest.approx(anchor.target_height - anchor.target_diameter)
            else:
                assert cfg.kwargs["height"] == pytest.approx(anchor.target_height)


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
