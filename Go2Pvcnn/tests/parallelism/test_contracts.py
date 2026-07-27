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
