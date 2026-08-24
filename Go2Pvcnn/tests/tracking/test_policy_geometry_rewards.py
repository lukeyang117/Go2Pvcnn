from types import SimpleNamespace

import torch

import tracking.mdp.policy_geometry_rewards as rewards


def _scan(batch: int = 2, side: int = 5, resolution: float = 0.1) -> torch.Tensor:
    axis = torch.arange(side, dtype=torch.float32) * resolution
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xyz = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)
    return xyz.reshape(1, side * side, 3).expand(batch, -1, -1).clone()


def _joint_names() -> tuple[str, ...]:
    return (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )


def test_parallelism_terrain_from_scan_preserves_grid_pose():
    hits = _scan()
    semantic = torch.zeros(2, 5, 5, dtype=torch.long)
    terrain = rewards.parallelism_terrain_from_scan(hits, semantic, None, resolution=0.1)

    assert terrain.height_w.shape == (2, 5, 5)
    assert terrain.semantic_id.shape == (2, 5, 5)
    assert terrain.valid_mask.all()
    assert torch.allclose(terrain.origin_w[:, :2], hits[:, 0, :2])
    assert torch.allclose(terrain.yaw_w, torch.zeros(2))
    assert terrain.resolution == 0.1


def test_live_policy_collision_aggregates_all_legs(monkeypatch):
    bits = torch.zeros(2, 4, 1, 6, dtype=torch.bool)
    bits[1, 2, 0, 1] = True
    monkeypatch.setattr(
        rewards,
        "official_collision_mask",
        lambda terrain, geometry, cfg: (~bits.any(-1), bits),
    )

    event = rewards.live_policy_geometry_collision_event(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]]),
        root_quat_w=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        joint_pos=torch.zeros(2, 12),
        joint_names=_joint_names(),
        terrain=rewards.parallelism_terrain_from_scan(
            _scan(), torch.zeros(2, 5, 5), None, resolution=0.1
        ),
    )

    assert event.tolist() == [0.0, 1.0]


def test_policy_reward_does_not_require_reference_manager(monkeypatch):
    monkeypatch.setattr(
        rewards,
        "live_policy_geometry_collision_event",
        lambda **kwargs: torch.tensor([0.0, 1.0]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(2, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
            joint_pos=torch.zeros(2, 12),
        ),
        joint_names=_joint_names(),
    )
    scanner = SimpleNamespace(
        cfg=SimpleNamespace(pattern_cfg=SimpleNamespace(resolution=0.1)),
        data=SimpleNamespace(
            ray_hits_w=_scan(),
            semantic_map=torch.zeros(2, 5, 5),
            valid_mask=None,
        ),
    )
    env = SimpleNamespace(
        scene={"robot": robot, "semantic_height_scanner": scanner}
    )

    result = rewards.policy_geometry_collision_penalty(
        env,
        asset_cfg=SimpleNamespace(name="robot"),
        scanner_cfg=SimpleNamespace(name="semantic_height_scanner"),
    )

    assert result.tolist() == [0.0, 1.0]


def test_policy_reward_source_has_no_reference_manager_dependency():
    from pathlib import Path

    source = Path("Go2Pvcnn/tracking/mdp/policy_geometry_rewards.py").read_text()
    assert "get_parallelism_reference_manager" not in source
    assert "tracking.managers" not in source
