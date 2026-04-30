from __future__ import annotations

import math
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
    DEFAULT_CENTER_SAFETY_HALF_EXTENT_M,
    DEFAULT_GROUNDING_EMBED_DEPTH_M,
    DEFAULT_GROUNDING_HEIGHT_QUANTILE,
    DEFAULT_MAX_LAYOUT_ATTEMPTS,
    DEFAULT_MIN_SPACING_CLEARANCE_M,
    DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG,
    DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG,
    DEFAULT_SEMANTIC_COURSE_SEED,
    DEFAULT_SEMANTIC_COURSE_TILE_SIZE,
    DEFAULT_TILE_MARGIN_M,
    DEFAULT_VIEWER_REPRESENTATIVE_STAGE,
    LARGE_OBSTACLE_SIZE,
    SEMANTIC_COURSE_LARGE_ROOT,
    SEMANTIC_COURSE_SMALL_ROOT,
    SHAPE_AXIS_Z,
    SHARED_NATIVE_SHAPE_POOL,
    SMALL_OBSTACLE_SIZE,
    SemanticCourseStage,
    SemanticCourseGroundingCfg,
    bottom_to_center_offset,
    build_course_anchors,
    course_anchor_counts,
    deterministic_shape_key,
    ground_course_anchors,
    footprint_sample_offsets,
    representative_rows,
    resolve_tile_size,
    select_shape_kind,
    shape_params_for_profile,
    semantic_scale_profile,
    set_scene_env_to_representative_stage,
    stage_for_row,
    _ground_with_runtime_terrain_sampler,
    _shape_spawn_cfg,
)


def _terrain_origins(num_rows: int = 10, num_cols: int = 2):
    origins = []
    for row in range(num_rows):
        origins.append([])
        for col in range(num_cols):
            origins[row].append((row * 8.0, col * 8.0, row * 0.1 + col * 0.01))
    return origins


def _s4_anchors(anchors):
    return [anchor for anchor in anchors if anchor.stage is SemanticCourseStage.S4]


def _assert_inside_margin(anchor, tile_size=DEFAULT_SEMANTIC_COURSE_TILE_SIZE):
    half_x = tile_size[0] / 2.0
    half_y = tile_size[1] / 2.0
    radius = anchor.target_diameter / 2.0
    assert -half_x + DEFAULT_TILE_MARGIN_M + radius <= anchor.local_xy[0] <= half_x - DEFAULT_TILE_MARGIN_M - radius
    assert -half_y + DEFAULT_TILE_MARGIN_M + radius <= anchor.local_xy[1] <= half_y - DEFAULT_TILE_MARGIN_M - radius


def _assert_outside_center_safety(anchor):
    x, y = anchor.local_xy
    assert abs(x) > DEFAULT_CENTER_SAFETY_HALF_EXTENT_M or abs(y) > DEFAULT_CENTER_SAFETY_HALF_EXTENT_M


def test_semantic_course_defaults_and_configs_are_importable_and_exact():
    assert DEFAULT_SEMANTIC_COURSE_SEED == 20260430
    assert DEFAULT_SEMANTIC_COURSE_TILE_SIZE == (8.0, 8.0)
    assert DEFAULT_TILE_MARGIN_M == pytest.approx(0.50)
    assert DEFAULT_CENTER_SAFETY_HALF_EXTENT_M == pytest.approx(0.85)
    assert DEFAULT_MIN_SPACING_CLEARANCE_M == pytest.approx(0.15)
    assert DEFAULT_MAX_LAYOUT_ATTEMPTS == 64
    assert DEFAULT_GROUNDING_HEIGHT_QUANTILE == pytest.approx(1.0)
    assert DEFAULT_GROUNDING_EMBED_DEPTH_M == pytest.approx(0.015)

    assert DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG.tile_margin_m == pytest.approx(DEFAULT_TILE_MARGIN_M)
    assert DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG.center_safety_half_extent_m == pytest.approx(
        DEFAULT_CENTER_SAFETY_HALF_EXTENT_M
    )
    assert DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG.min_spacing_clearance_m == pytest.approx(
        DEFAULT_MIN_SPACING_CLEARANCE_M
    )
    assert DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG.max_layout_attempts == DEFAULT_MAX_LAYOUT_ATTEMPTS
    assert DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG.height_quantile == pytest.approx(
        DEFAULT_GROUNDING_HEIGHT_QUANTILE
    )
    assert DEFAULT_SEMANTIC_COURSE_GROUNDING_CFG.embed_depth_m == pytest.approx(DEFAULT_GROUNDING_EMBED_DEPTH_M)


def test_resolve_tile_size_prefers_terrain_generator_size_before_origin_inference_and_fallback():
    class TerrainGenerator:
        size = (6.0, 7.0)

    origins = _terrain_origins(num_rows=3, num_cols=3)
    assert resolve_tile_size(origins, terrain_generator=TerrainGenerator()) == (6.0, 7.0)
    assert resolve_tile_size([[(0.0, 0.0, 0.0)]], fallback_tile_size=(5.0, 4.0)) == (5.0, 4.0)


def test_resolve_tile_size_infers_x_y_spacing_from_terrain_origins_without_generator():
    origins = [
        [(0.0, 0.0, 0.0), (0.0, 7.0, 0.0), (0.0, 14.0, 0.0)],
        [(5.0, 0.0, 0.0), (5.0, 7.0, 0.0), (5.0, 14.0, 0.0)],
        [(10.0, 0.0, 0.0), (10.0, 7.0, 0.0), (10.0, 14.0, 0.0)],
    ]
    assert resolve_tile_size(origins) == (5.0, 7.0)

    one_axis = [[(0.0, 0.0, 0.0)], [(6.5, 0.0, 0.0)], [(13.0, 0.0, 0.0)]]
    assert resolve_tile_size(one_axis, fallback_tile_size=(8.0, 9.0)) == (6.5, 9.0)


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
    assert all(hasattr(anchor, "layout_fallback_used") for anchor in anchors)


def test_build_course_anchors_is_reproducible_with_same_seed():
    terrain_origins = _terrain_origins(num_rows=10, num_cols=2)
    anchors_a = build_course_anchors(terrain_origins, semantic_course_seed=12345)
    anchors_b = build_course_anchors(terrain_origins, semantic_course_seed=12345)

    assert anchors_a == anchors_b


def test_different_row_col_layouts_differ():
    anchors = build_course_anchors(_terrain_origins(num_rows=10, num_cols=2))

    row8_col0 = [(a.semantic_class, a.slot_index, a.local_xy) for a in _s4_anchors(anchors) if a.row == 8 and a.col == 0]
    row8_col1 = [(a.semantic_class, a.slot_index, a.local_xy) for a in _s4_anchors(anchors) if a.row == 8 and a.col == 1]
    row9_col0 = [(a.semantic_class, a.slot_index, a.local_xy) for a in _s4_anchors(anchors) if a.row == 9 and a.col == 0]

    assert row8_col0 != row8_col1
    assert row8_col0 != row9_col0


def test_default_local_points_stay_inside_tile_margins_and_avoid_center_safety_box():
    anchors = build_course_anchors(_terrain_origins(num_rows=10, num_cols=2))

    for anchor in anchors:
        _assert_inside_margin(anchor)
        _assert_outside_center_safety(anchor)


def test_object_spacing_constraints_hold_for_default_non_fallback_anchors():
    anchors = [
        anchor
        for anchor in build_course_anchors(_terrain_origins(num_rows=10, num_cols=1))
        if anchor.row == 8 and anchor.col == 0
    ]
    assert anchors
    assert not any(anchor.layout_fallback_used for anchor in anchors)

    for index, anchor_a in enumerate(anchors):
        for anchor_b in anchors[index + 1 :]:
            dx = anchor_a.local_xy[0] - anchor_b.local_xy[0]
            dy = anchor_a.local_xy[1] - anchor_b.local_xy[1]
            distance = math.hypot(dx, dy)
            expected = (
                anchor_a.target_diameter / 2.0
                + anchor_b.target_diameter / 2.0
                + DEFAULT_MIN_SPACING_CLEARANCE_M
            )
            assert distance + 1.0e-9 >= expected


def test_canonical_s4_default_layout_does_not_use_fallback():
    anchors = [
        anchor
        for anchor in build_course_anchors(_terrain_origins(num_rows=10, num_cols=1), tile_size=(8.0, 8.0))
        if anchor.row == 8 and anchor.col == 0
    ]

    assert len(anchors) == 7
    assert not any(anchor.layout_fallback_used for anchor in anchors)


def test_tight_tile_can_use_fallback_and_still_satisfies_margin_and_center_safety():
    tile_size = (3.4, 3.4)
    tight_layout_cfg = type(DEFAULT_SEMANTIC_COURSE_LAYOUT_CFG)(
        tile_margin_m=DEFAULT_TILE_MARGIN_M,
        center_safety_half_extent_m=DEFAULT_CENTER_SAFETY_HALF_EXTENT_M,
        min_spacing_clearance_m=DEFAULT_MIN_SPACING_CLEARANCE_M,
        max_layout_attempts=1,
    )
    anchors = [
        anchor
        for anchor in build_course_anchors(
            _terrain_origins(num_rows=10, num_cols=1),
            tile_size=tile_size,
            layout_cfg=tight_layout_cfg,
        )
        if anchor.row == 8 and anchor.col == 0
    ]

    assert len(anchors) == 7
    assert any(anchor.layout_fallback_used for anchor in anchors)
    for anchor in anchors:
        _assert_inside_margin(anchor, tile_size=tile_size)
        _assert_outside_center_safety(anchor)


def test_default_s4_anchors_spread_outside_old_center_scanner_footprint():
    anchors = [
        anchor
        for anchor in build_course_anchors(_terrain_origins(num_rows=10, num_cols=1))
        if anchor.row == 8 and anchor.col == 0
    ]

    assert anchors
    assert any(abs(anchor.local_xy[0]) > 0.75 or abs(anchor.local_xy[1]) > 0.75 for anchor in anchors)


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


@pytest.mark.parametrize("shape_kind", SHARED_NATIVE_SHAPE_POOL)
def test_footprint_sample_offsets_cover_center_and_eight_support_points(shape_kind):
    shape_params = shape_params_for_profile(
        shape_kind,
        target_diameter=0.45,
        target_height=0.55,
    )

    offsets = footprint_sample_offsets(shape_kind, shape_params)

    assert len(offsets) == 9
    assert (0.0, 0.0) in offsets
    assert len(set(offsets)) == 9


def test_ground_course_anchors_uses_footprint_max_height_not_center_height():
    terrain_origins = _terrain_origins(num_rows=10, num_cols=1)
    anchors = build_course_anchors(terrain_origins)
    selected = [next(anchor for anchor in anchors if anchor.shape_kind == "cuboid")]

    grounded = ground_course_anchors(
        selected,
        terrain_height_at_xy=lambda x, y: 10.0
        if (x, y) == selected[0].world_xy
        else 10.4,
    )

    obstacle = grounded[0]
    expected_z = 10.4 - DEFAULT_GROUNDING_EMBED_DEPTH_M + selected[0].ground_offset
    assert obstacle.world_center[:2] == selected[0].world_xy
    assert obstacle.world_center[2] == pytest.approx(expected_z)
    assert obstacle.shape_kind == selected[0].shape_kind
    assert obstacle.shape_params == selected[0].shape_params


@pytest.mark.parametrize("shape_kind", SHARED_NATIVE_SHAPE_POOL)
def test_ground_course_anchors_default_formula_is_max_footprint_minus_embed_plus_shape_offset(shape_kind):
    anchor = next(anchor for anchor in build_course_anchors(_terrain_origins()) if anchor.shape_kind == shape_kind)
    footprint = footprint_sample_offsets(anchor.shape_kind, anchor.shape_params)
    heights_by_xy = {
        (anchor.world_xy[0] + dx, anchor.world_xy[1] + dy): 0.1 * index
        for index, (dx, dy) in enumerate(footprint)
    }

    grounded = ground_course_anchors(
        [anchor],
        terrain_height_at_xy=lambda x, y: heights_by_xy[(x, y)],
    )

    expected_z = max(heights_by_xy.values()) - DEFAULT_GROUNDING_EMBED_DEPTH_M + anchor.ground_offset
    assert grounded[0].world_center[2] == pytest.approx(expected_z)


def test_ground_course_anchors_rejects_non_finite_footprint_heights():
    anchor = build_course_anchors(_terrain_origins())[0]

    with pytest.raises(ValueError, match="non-finite terrain height"):
        ground_course_anchors([anchor], terrain_height_at_xy=lambda *_: math.nan)


def test_ground_course_anchors_rejects_invalid_embed_depth():
    anchor = build_course_anchors(_terrain_origins())[0]

    with pytest.raises(ValueError, match="grounding embed_depth_m"):
        ground_course_anchors(
            [anchor],
            terrain_height_at_xy=lambda *_: 0.0,
            grounding_cfg=SemanticCourseGroundingCfg(embed_depth_m=-0.001),
        )


def test_runtime_grounding_batches_footprints_and_honors_grounding_cfg(monkeypatch):
    anchors = build_course_anchors(_terrain_origins())[:2]
    captured_xy_points = []

    def fake_sample_terrain_heights_world(xy_points, *, device):
        assert device == "cuda:0"
        captured_xy_points.extend(xy_points)
        heights = []
        for anchor_index, _anchor in enumerate(anchors):
            heights.extend([anchor_index + index * 0.1 for index in range(9)])
        return heights

    monkeypatch.setattr(
        "extension.semantic_course._sample_terrain_heights_world",
        fake_sample_terrain_heights_world,
    )

    obstacles = _ground_with_runtime_terrain_sampler(
        anchors,
        device="cuda:0",
        grounding_cfg=SemanticCourseGroundingCfg(height_quantile=0.5, embed_depth_m=0.02),
    )

    expected_xy_points = [
        (anchor.world_xy[0] + dx, anchor.world_xy[1] + dy)
        for anchor in anchors
        for dx, dy in footprint_sample_offsets(anchor.shape_kind, anchor.shape_params)
    ]
    assert captured_xy_points == expected_xy_points
    assert len(captured_xy_points) == 9 * len(anchors)
    assert obstacles[0].world_center[2] == pytest.approx(0.4 - 0.02 + anchors[0].ground_offset)
    assert obstacles[1].world_center[2] == pytest.approx(1.4 - 0.02 + anchors[1].ground_offset)


def test_uneven_fake_terrain_embeds_slightly_at_high_footprint_point():
    anchor = next(anchor for anchor in build_course_anchors(_terrain_origins()) if anchor.shape_kind == "sphere")
    center_x, center_y = anchor.world_xy
    high_point = max(
        ((center_x + dx, center_y + dy) for dx, dy in footprint_sample_offsets(anchor.shape_kind, anchor.shape_params)),
        key=lambda xy: xy[0],
    )

    def fake_terrain(x, y):
        if (x, y) == high_point:
            return 0.5
        if (x, y) == anchor.world_xy:
            return 0.1
        return 0.2

    obstacle = ground_course_anchors([anchor], terrain_height_at_xy=fake_terrain)[0]
    bottom_z = obstacle.world_center[2] - anchor.ground_offset

    assert bottom_z == pytest.approx(0.5 - DEFAULT_GROUNDING_EMBED_DEPTH_M)
    assert bottom_z > 0.1


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
