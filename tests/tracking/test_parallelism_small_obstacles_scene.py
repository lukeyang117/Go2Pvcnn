from __future__ import annotations

import math
import types


def test_fixed_scene_has_one_subterrain_and_circular_reset_hole(monkeypatch) -> None:
    from tracking.parallelism_small_obstacles_scene import (
        ParallelismSmallObstacleSceneCfg,
        build_small_obstacle_local_xy,
        small_obstacles_terrain_cfg,
    )

    class TerrainGeneratorCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class MeshPlaneTerrainCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "isaaclab.terrains",
        types.SimpleNamespace(
            terrain_gen=types.SimpleNamespace(
                TerrainGeneratorCfg=TerrainGeneratorCfg,
                MeshPlaneTerrainCfg=MeshPlaneTerrainCfg,
            )
        ),
    )

    cfg = ParallelismSmallObstacleSceneCfg()
    points = build_small_obstacle_local_xy(cfg)
    terrain_cfg = small_obstacles_terrain_cfg(cfg)

    assert len(points) == cfg.small_obstacle_count == 24
    assert terrain_cfg.num_rows == 1
    assert terrain_cfg.num_cols == 1
    assert tuple(terrain_cfg.sub_terrains) == ("small_obstacles",)

    half_patch = cfg.obstacle_patch_size_m / 2.0
    for x, y in points:
        assert -half_patch <= x <= half_patch
        assert -half_patch <= y <= half_patch
        assert math.hypot(x, y) >= cfg.obstacle_center_exclusion_radius_m


def test_fixed_scene_layout_is_deterministic_and_respects_spacing() -> None:
    from tracking.parallelism_small_obstacles_scene import (
        ParallelismSmallObstacleSceneCfg,
        build_small_obstacle_local_xy,
    )

    cfg = ParallelismSmallObstacleSceneCfg()
    first = build_small_obstacle_local_xy(cfg)
    second = build_small_obstacle_local_xy(cfg)

    assert first == second
    for index, (x0, y0) in enumerate(first):
        for x1, y1 in first[index + 1 :]:
            assert math.hypot(x0 - x1, y0 - y1) >= cfg.small_obstacle_min_spacing_m
