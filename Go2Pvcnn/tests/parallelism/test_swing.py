from __future__ import annotations

import torch


def _terrain_with_mid_obstacle():
    from extension.parallelism import ParallelismTerrain

    height = torch.zeros(1, 21, 21, dtype=torch.float32)
    height[:, 5, 10] = 0.16
    return ParallelismTerrain(
        height_w=height,
        semantic_id=torch.zeros(1, 21, 21, dtype=torch.long),
        valid_mask=torch.ones(1, 21, 21, dtype=torch.bool),
        origin_w=torch.tensor([[-0.5, -0.5, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )


def test_terrain_aware_swing_keeps_parabola_endpoints_and_clears_path_height():
    from extension.parallelism.swing import terrain_aware_swing_curve

    terrain = _terrain_with_mid_obstacle()
    start = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    touchdown = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)

    swing = terrain_aware_swing_curve(
        start,
        touchdown,
        terrain,
        frames=11,
        clearance_m=0.03,
        min_apex_m=0.08,
    )

    assert swing.shape == (1, 1, 11, 3)
    assert torch.allclose(swing[:, :, 0], start)
    assert torch.allclose(swing[:, :, -1], touchdown)
    assert torch.isclose(swing[0, 0, 5, 2], torch.tensor(0.19), atol=1e-5)

    tau = torch.linspace(0.0, 1.0, 11)
    shape = 4.0 * tau * (1.0 - tau)
    apex = torch.where(
        shape > 0.0,
        swing[0, 0, :, 2] / shape.clamp_min(1e-6),
        torch.zeros_like(shape),
    )
    assert torch.allclose(apex[1:-1], torch.full((9,), apex[5]), atol=1e-5)


def test_terrain_aware_swing_uses_min_apex_on_flat_ground():
    from extension.parallelism import ParallelismTerrain
    from extension.parallelism.swing import terrain_aware_swing_curve

    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 7, 7),
        semantic_id=torch.zeros(1, 7, 7, dtype=torch.long),
        valid_mask=torch.ones(1, 7, 7, dtype=torch.bool),
        origin_w=torch.tensor([[-0.3, -0.3, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    start = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    touchdown = torch.tensor([[[0.6, 0.0, 0.0]]], dtype=torch.float32)

    swing = terrain_aware_swing_curve(
        start,
        touchdown,
        terrain,
        frames=7,
        clearance_m=0.0,
        min_apex_m=0.08,
    )

    assert torch.isclose(swing[0, 0, 3, 2], torch.tensor(0.08), atol=1e-6)


def test_parallelism_cfg_exposes_swing_clearance_and_min_apex():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()

    assert cfg.swing_clearance_m == 0.05
    assert cfg.min_swing_apex_m == 0.08
