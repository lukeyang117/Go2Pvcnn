import torch

from extension.parallelism.config import ParallelismCfg
from extension.parallelism.root import rollout_root, soft_clamp_terrain_command
from extension.parallelism.types import ParallelismState, ParallelismTerrain


def _terrain_from_height(height: torch.Tensor, *, resolution: float = 0.01) -> ParallelismTerrain:
    batch = int(height.shape[0])
    side = int(height.shape[-1])
    origin = torch.zeros(batch, 3, dtype=torch.float32)
    origin[:, 0] = -0.5 * float(side - 1) * resolution
    origin[:, 1] = -0.5 * float(side - 1) * resolution
    return ParallelismTerrain(
        height_w=height.to(dtype=torch.float32),
        semantic_id=torch.zeros_like(height, dtype=torch.long),
        valid_mask=torch.ones_like(height, dtype=torch.bool),
        origin_w=origin,
        yaw_w=torch.zeros(batch, dtype=torch.float32),
        resolution=resolution,
    )


def _state(root_z: float = 0.42) -> ParallelismState:
    foot = torch.tensor(
        [
            [
                [0.20, 0.12, 0.0],
                [0.20, -0.12, 0.0],
                [-0.20, 0.12, 0.0],
                [-0.20, -0.12, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    return ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, root_z]], dtype=torch.float32),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
        foot_pos_w=foot,
    )


def test_flat_mask_keeps_stance_foot_root_height_rule():
    height = torch.full((1, 151, 151), 0.20)
    terrain = _terrain_from_height(height)
    cfg = ParallelismCfg(root_clearance_m=0.30)

    root = rollout_root(
        _state(root_z=0.42),
        torch.zeros(1, 3),
        terrain,
        cfg,
        terrain_following_mask=torch.tensor([False]),
    )

    assert torch.allclose(root.root_pos_w[..., 2], torch.full((1, cfg.horizon), 0.50))


def test_nonflat_mask_uses_height_map_at_root_xy_and_keeps_first_frame_real_root():
    side = 151
    x = (torch.arange(side, dtype=torch.float32) - 75.0) * 0.01
    height = (0.10 + 0.25 * x).view(1, 1, side).expand(1, side, side).clone()
    terrain = _terrain_from_height(height)
    cfg = ParallelismCfg(
        root_clearance_m=0.30,
        terrain_following_root_clearance_m=0.30,
        terrain_following_root_z_smoothing=1.0,
        terrain_following_root_z_rate_limit_m=10.0,
        terrain_following_root_height_deadband_m=0.0,
    )

    root = rollout_root(
        _state(root_z=0.42),
        torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
        terrain,
        cfg,
        terrain_following_mask=torch.tensor([True]),
    )

    assert torch.allclose(root.root_pos_w[:, 0, 2], torch.tensor([0.42]))
    assert root.root_pos_w[0, -1, 2] > root.root_pos_w[0, 1, 2]
    assert torch.allclose(root.root_pos_w[0, 1, 2], torch.tensor(0.10 + 0.30), atol=0.015)


def test_nonflat_soft_clamp_scales_only_excess_velocity():
    cfg = ParallelismCfg(
        terrain_following_vx_soft_limit=0.5,
        terrain_following_vy_soft_limit=0.25,
        terrain_following_vyaw_soft_limit=0.5,
        terrain_following_vx_excess_scale=0.5,
        terrain_following_vy_excess_scale=0.5,
        terrain_following_vyaw_excess_scale=0.5,
    )

    command = torch.tensor([[1.0, -0.5, 1.0], [0.4, 0.2, -0.4]], dtype=torch.float32)
    clamped = soft_clamp_terrain_command(command, cfg)

    assert torch.allclose(clamped[0], torch.tensor([0.75, -0.375, 0.75]))
    assert torch.allclose(clamped[1], command[1])


def test_nonflat_roll_pitch_follow_local_height_difference_with_limits():
    side = 151
    x = (torch.arange(side, dtype=torch.float32) - 75.0) * 0.01
    height = (0.20 * x).view(1, 1, side).expand(1, side, side).clone()
    terrain = _terrain_from_height(height)
    cfg = ParallelismCfg(
        terrain_following_root_clearance_m=0.30,
        terrain_following_root_z_smoothing=1.0,
        terrain_following_root_z_rate_limit_m=10.0,
        terrain_following_rpy_smoothing=1.0,
        terrain_following_pitch_limit_rad=0.08,
        terrain_following_roll_limit_rad=0.08,
        terrain_following_rpy_rate_limit_rad=10.0,
    )

    root = rollout_root(
        _state(root_z=0.42),
        torch.zeros(1, 3),
        terrain,
        cfg,
        terrain_following_mask=torch.tensor([True]),
    )

    assert torch.allclose(root.root_rpy_w[:, 0, :2], torch.zeros(1, 2))
    assert root.root_rpy_w[0, 1:, 1].abs().amax() <= 0.0801
    assert torch.all(root.root_rpy_w[0, 1:, 1] < 0.0)
