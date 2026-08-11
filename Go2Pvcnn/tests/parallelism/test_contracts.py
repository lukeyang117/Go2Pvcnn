from __future__ import annotations

import importlib
import sys

import torch


def test_parallelism_import_isolation():
    before = set(sys.modules)
    module = importlib.import_module("extension.parallelism")
    after = set(sys.modules)
    newly_loaded = after - before
    forbidden = {
        name
        for name in newly_loaded
        if name.startswith("extension.joint_mpc_rti")
        or name.startswith("extension.batch_mpc_planner")
    }
    assert forbidden == set()
    assert hasattr(module, "ParallelismCfg")


def test_terrain_query_batched_points():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.terrain import query_height_semantic_valid

    height = torch.arange(25, dtype=torch.float32).reshape(1, 5, 5)
    semantic = torch.full((1, 5, 5), 7, dtype=torch.long)
    valid = torch.ones((1, 5, 5), dtype=torch.bool)
    terrain = ParallelismTerrain(
        height_w=height,
        semantic_id=semantic,
        valid_mask=valid,
        origin_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    points = torch.tensor([[[0.0, 0.0], [0.2, 0.2], [1.0, 1.0]]], dtype=torch.float32)
    result = query_height_semantic_valid(terrain, points)

    assert result.height.shape == (1, 3)
    assert result.semantic.shape == (1, 3)
    assert result.valid.tolist() == [[True, True, False]]
    assert result.semantic[0, 0].item() == 7


def test_parallelism_cfg_exposes_semantic_touchdown_margin():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()

    assert cfg.semantic_touchdown_margin_m == 0.12


def test_parallelism_cfg_exposes_large_obstacle_avoidance_parameters():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()

    assert cfg.large_obstacle_rect_width_m == 0.70
    assert cfg.large_obstacle_rect_length_m == 1.20
    assert cfg.large_obstacle_lateral_speed_max_mps == 0.25
    assert cfg.large_obstacle_default_side == 1  # +1=left, -1=right


def test_expanded_obstacle_mask_blocks_neighboring_touchdown():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.terrain import expanded_obstacle_mask, query_expanded_obstacle

    semantic = torch.zeros(1, 5, 5, dtype=torch.long)
    semantic[:, 2, 2] = 1
    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 5, 5),
        semantic_id=semantic,
        valid_mask=torch.ones(1, 5, 5, dtype=torch.bool),
        origin_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.01,
    )

    mask = expanded_obstacle_mask(terrain, (1, 2), margin_m=0.01)
    points = torch.tensor([[[0.01, 0.02], [0.04, 0.04]]], dtype=torch.float32)
    blocked = query_expanded_obstacle(terrain, points, mask)

    assert blocked.tolist() == [[True, False]]
