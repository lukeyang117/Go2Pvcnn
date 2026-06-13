from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))


from extension.mdp.semantic_body_part_clearance import (
    _body_geometry_query_points,
    _cached_circle_offsets,
    _current_body_part_sample_points,
    _semantic_clearance_penalty_from_points,
    _semantic_geometry_clearance_penalty,
    semantic_foot_over_clearance_bonus_from_tensors,
)
from extension.batch_mpc_planner.terrain import MpcPlannerTerrain


def _maps(*, semantic_id: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    elevation = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    elevation[:, 2, 2] = 0.10
    semantic[:, 2, 2] = int(semantic_id)
    return elevation, semantic


def _points(z: float, *, x: float = 0.0, y: float = 0.0, part: str = "foot") -> dict[str, torch.Tensor]:
    return {part: torch.tensor([[[[x, y, z]]]], dtype=torch.float32)}


def _penalty(points_by_part: dict[str, torch.Tensor], *, semantic_id: int = 1) -> torch.Tensor:
    elevation, semantic = _maps(semantic_id=semantic_id)
    return _semantic_clearance_penalty_from_points(
        terrain=MpcPlannerTerrain(
            height_map=elevation,
            semantic_map=semantic,
            world_x_range=(-0.2, 0.2),
            world_y_range=(-0.2, 0.2),
        ),
        points_by_part=points_by_part,
        small_semantic_ids=(1,),
        margins={"foot": 0.02, "shank": 0.04, "thigh": 0.04},
        weights={"foot": 0.5, "shank": 2.0, "thigh": 2.0},
        penalty_clip=1.0,
    )


def test_no_small_semantic_cells_returns_zero_penalty() -> None:
    reward = _penalty(_points(0.0), semantic_id=0)

    torch.testing.assert_close(reward, torch.zeros(1))


def test_small_semantic_deficit_returns_negative_reward() -> None:
    reward = _penalty(_points(0.0))

    assert reward.shape == (1,)
    assert reward.item() < 0.0


def test_shank_and_thigh_weights_exceed_foot_for_same_deficit() -> None:
    foot_reward = _penalty(_points(0.0, part="foot"))
    shank_reward = _penalty(_points(0.0, part="shank"))
    thigh_reward = _penalty(_points(0.0, part="thigh"))

    assert shank_reward.item() < foot_reward.item()
    assert thigh_reward.item() < foot_reward.item()
    torch.testing.assert_close(shank_reward, thigh_reward)


def test_ground_semantic_id_has_no_penalty_even_when_height_close() -> None:
    reward = _penalty(_points(0.0), semantic_id=0)

    assert reward.item() == 0.0


def test_foot_sphere_neighborhood_detects_adjacent_small_cell() -> None:
    elevation = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    elevation[:, 4, 5] = 0.10
    semantic[:, 4, 5] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.08, 0.08),
        world_y_range=(-0.08, 0.08),
    )
    centers = {"foot": torch.tensor([[[[0.0, 0.0, 0.11]]]], dtype=torch.float32)}

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part=centers,
        root_pos_w=torch.tensor([[0.0, 0.0, 0.20]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        foot_radius_m=0.022,
        foot_query_radius_m=0.035,
        foot_margin_m=0.015,
        foot_weight=0.5,
        penalty_clip=1.0,
    )

    assert reward.shape == (1,)
    assert reward.item() < 0.0


def test_calf_capsule_neighborhood_detects_adjacent_small_cell() -> None:
    elevation = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    elevation[:, 4, 5] = 0.16
    semantic[:, 4, 5] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.08, 0.08),
        world_y_range=(-0.08, 0.08),
    )
    calf = torch.zeros((1, 4, 7, 3), dtype=torch.float32)
    calf[..., 2] = 0.18

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={"calf": calf},
        root_pos_w=torch.tensor([[0.0, 0.0, 0.25]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        calf_radius_m=0.040,
        calf_query_radius_m=0.045,
        calf_margin_m=0.040,
        calf_weight=2.0,
        penalty_clip=1.0,
    )

    assert torch.isfinite(reward).all()
    assert reward.item() < 0.0


def test_thigh_capsule_neighborhood_detects_adjacent_small_cell() -> None:
    elevation = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    elevation[:, 5, 4] = 0.20
    semantic[:, 5, 4] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.08, 0.08),
        world_y_range=(-0.08, 0.08),
    )
    thigh = torch.zeros((1, 4, 7, 3), dtype=torch.float32)
    thigh[..., 2] = 0.23

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={"thigh": thigh},
        root_pos_w=torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        thigh_radius_m=0.040,
        thigh_query_radius_m=0.045,
        thigh_margin_m=0.040,
        thigh_weight=1.5,
        penalty_clip=1.0,
    )

    assert torch.isfinite(reward).all()
    assert reward.item() < 0.0


def test_base_footprint_grid_detects_small_cell_under_body_extent() -> None:
    elevation = torch.zeros((1, 17, 17), dtype=torch.float32)
    semantic = torch.zeros((1, 17, 17), dtype=torch.long)
    elevation[:, 8, 12] = 0.14
    semantic[:, 8, 12] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.24, 0.24),
        world_y_range=(-0.24, 0.24),
    )

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={},
        root_pos_w=torch.tensor([[0.0, 0.0, 0.18]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        small_semantic_ids=(1,),
        include_base=True,
        base_half_extents_m=(0.20, 0.06, 0.07),
        base_footprint_grid=(5, 3),
        base_query_radius_m=0.030,
        base_margin_m=0.020,
        base_weight=1.0,
        penalty_clip=1.0,
    )

    assert reward.shape == (1,)
    assert reward.item() < 0.0


def test_geometry_clearance_returns_fixed_shape_for_batched_envs() -> None:
    num_envs = 128
    elevation = torch.zeros((num_envs, 17, 17), dtype=torch.float32)
    semantic = torch.zeros((num_envs, 17, 17), dtype=torch.long)
    elevation[:, 8, 8] = 0.10
    semantic[:, 8, 8] = 1
    terrain = MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-0.24, 0.24),
        world_y_range=(-0.24, 0.24),
    )
    foot = torch.zeros((num_envs, 4, 1, 3), dtype=torch.float32)
    foot[..., 2] = 0.11

    reward = _semantic_geometry_clearance_penalty(
        terrain=terrain,
        centers_by_part={"foot": foot},
        root_pos_w=torch.zeros((num_envs, 3), dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_envs, 1),
        small_semantic_ids=(1,),
    )

    assert reward.shape == (num_envs,)
    assert torch.isfinite(reward).all()


def test_large_query_radius_uses_more_offsets_for_signal_probe() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 17, 17), dtype=torch.float32),
        semantic_map=torch.zeros((1, 17, 17), dtype=torch.long),
        world_x_range=(-0.24, 0.24),
        world_y_range=(-0.24, 0.24),
    )
    centers = torch.zeros((1, 1, 3), dtype=torch.float32)
    surface_z = torch.zeros((1, 1), dtype=torch.float32)

    query_xy, query_surface_z = _body_geometry_query_points(
        centers=centers,
        surface_z=surface_z,
        query_radius_m=0.12,
        terrain=terrain,
    )

    assert query_xy.shape[1] > 13
    assert query_surface_z.shape == query_xy.shape[:2]


def test_geometry_reward_hot_path_has_no_per_env_loop() -> None:
    source = (GO2PVCNN_ROOT / "extension/mdp/semantic_body_part_clearance.py").read_text(encoding="utf-8")
    forbidden = ["for env_id in", "for env_idx in", "range(num_envs)", "range(env.num_envs)"]
    for text in forbidden:
        assert text not in source


def test_foot_over_bonus_rewards_clear_foot_above_path_small_cell() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.12, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32),
        semantic_map=torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=torch.long),
        world_x_range=(-0.2, 0.2),
        world_y_range=(-0.2, 0.2),
    )
    root_pos = torch.tensor([[-0.20, 0.0, 0.30]], dtype=torch.float32)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    command = torch.tensor([[0.8, 0.0, 0.0]], dtype=torch.float32)
    clear_foot = torch.tensor([[[0.0, 0.0, 0.24], [0.15, 0.18, 0.02], [0.15, -0.18, 0.02], [-0.15, 0.0, 0.02]]])
    low_foot = clear_foot.clone()
    low_foot[:, 0, 2] = 0.14

    clear_bonus = semantic_foot_over_clearance_bonus_from_tensors(
        terrain=terrain,
        foot_pos_w=clear_foot,
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        command=command,
        small_semantic_ids=(1,),
        clearance_margin_m=0.05,
    )
    low_bonus = semantic_foot_over_clearance_bonus_from_tensors(
        terrain=terrain,
        foot_pos_w=low_foot,
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        command=command,
        small_semantic_ids=(1,),
        clearance_margin_m=0.05,
    )

    assert clear_bonus.item() > 0.0
    assert low_bonus.item() == pytest.approx(0.0)


def test_scanner_pose_projects_world_points_into_current_map_frame() -> None:
    elevation, semantic = _maps(semantic_id=1)
    reward = _semantic_clearance_penalty_from_points(
        terrain=MpcPlannerTerrain(
            height_map=elevation,
            semantic_map=semantic,
            world_x_range=(-0.2, 0.2),
            world_y_range=(-0.2, 0.2),
            sensor_pos_w=torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32),
            sensor_yaw=torch.tensor([torch.pi / 2.0], dtype=torch.float32),
        ),
        points_by_part={"foot": torch.tensor([[[[1.0, 2.0, 0.0]]]], dtype=torch.float32)},
        small_semantic_ids=(1,),
        margins={"foot": 0.02, "shank": 0.04, "thigh": 0.04},
        weights={"foot": 0.5, "shank": 2.0, "thigh": 2.0},
        penalty_clip=1.0,
    )

    assert reward.item() < 0.0


class _FakeRobot:
    def __init__(self, body_pos_w: torch.Tensor) -> None:
        root_pos_w = torch.zeros((body_pos_w.shape[0], 3), dtype=body_pos_w.dtype, device=body_pos_w.device)
        self.data = type(
            "Data",
            (),
            {
                "body_pos_w": body_pos_w,
                "root_pos_w": root_pos_w,
                "root_quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=body_pos_w.dtype).repeat(
                    body_pos_w.shape[0], 1
                ),
            },
        )()

    def find_bodies(self, pattern):
        names = {
            ".*_foot": ([2, 5, 8, 11], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            ".*_calf": ([1, 4, 7, 10], ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
            ".*_thigh": ([0, 3, 6, 9], ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]),
        }
        return names[pattern]


def test_current_body_part_sample_points_shapes_and_segments() -> None:
    body_pos = torch.zeros((2, 12, 3), dtype=torch.float32)
    body_ids = {"thigh": [0, 3, 6, 9], "calf": [1, 4, 7, 10], "foot": [2, 5, 8, 11]}
    for leg in range(4):
        thigh_id = body_ids["thigh"][leg]
        calf_id = body_ids["calf"][leg]
        foot_id = body_ids["foot"][leg]
        body_pos[:, thigh_id] = torch.tensor([float(leg), 0.0, 0.30])
        body_pos[:, calf_id] = torch.tensor([float(leg), 0.0, 0.15])
        body_pos[:, foot_id] = torch.tensor([float(leg), 0.0, 0.00])

    points = _current_body_part_sample_points(
        _FakeRobot(body_pos),
        body_ids=body_ids,
        shank_sample_count=2,
        thigh_sample_count=2,
    )

    assert points["foot"].shape == (2, 4, 1, 3)
    assert points["shank"].shape == (2, 4, 2, 3)
    assert points["thigh"].shape == (2, 4, 2, 3)
    torch.testing.assert_close(points["foot"][:, :, 0], body_pos[:, body_ids["foot"]])
    assert torch.all(points["shank"][..., 2] <= 0.15)
    assert torch.all(points["shank"][..., 2] >= 0.0)
    assert torch.all(points["thigh"][..., 2] <= 0.30)
    assert torch.all(points["thigh"][..., 2] >= 0.15)


class _FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.common_step_counter = 0
        self.device = "cpu"
        self.num_envs = 1


class _FakeScanner:
    def __init__(self) -> None:
        self.frame_count = 1
        self.data = type(
            "ScannerData",
            (),
            {
                "height_map": torch.zeros((1, 3, 3), dtype=torch.float32),
                "semantic_map": torch.zeros((1, 3, 3), dtype=torch.long),
                "pos_w": torch.zeros((1, 3), dtype=torch.float32),
                "quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            },
        )()


class _SceneEntity:
    def __init__(self, name: str, body_names=None) -> None:
        self.name = name
        self.body_names = body_names


class _FakeScene(dict):
    pass


class _FakeContactSensor:
    def __init__(self) -> None:
        self.data = type("ContactData", (), {"net_forces_w": torch.zeros((1, 4, 3), dtype=torch.float32)})()


def test_reward_wrapper_uses_current_robot_root_not_cached_anchor() -> None:
    from extension.mdp.semantic_body_part_clearance import semantic_body_part_clearance_reward

    body_pos = torch.zeros((1, 12, 3), dtype=torch.float32)
    body_pos[:, [0, 3, 6, 9], 2] = 0.30
    body_pos[:, [1, 4, 7, 10], 2] = 0.15
    body_pos[:, [2, 5, 8, 11], 2] = 0.00
    env = _FakeEnv()
    env.scene = _FakeScene(
        {
            "robot": _FakeRobot(body_pos),
            "semantic_height_scanner": _FakeScanner(),
            "contact_forces": _FakeContactSensor(),
        }
    )
    env.scene["semantic_height_scanner"].data.height_map[:, 1, 1] = 0.10
    env.scene["semantic_height_scanner"].data.semantic_map[:, 1, 1] = 1

    reward_before_move = semantic_body_part_clearance_reward(
        env,
        asset_cfg=_SceneEntity("robot"),
        scanner_cfg=_SceneEntity("semantic_height_scanner"),
        contact_sensor_cfg=_SceneEntity("contact_forces", body_names=".*_foot"),
    )
    env.scene["robot"].data.root_pos_w[:, 0] = 0.20
    env.scene["robot"].data.body_pos_w[:, :, 0] = 0.20
    reward_after_move = semantic_body_part_clearance_reward(
        env,
        asset_cfg=_SceneEntity("robot"),
        scanner_cfg=_SceneEntity("semantic_height_scanner"),
        contact_sensor_cfg=_SceneEntity("contact_forces", body_names=".*_foot"),
    )

    assert reward_before_move.item() < 0.0
    assert reward_after_move.item() < 0.0


def test_reward_wrapper_returns_finite_negative_values_for_body_part_deficit() -> None:
    from extension.mdp.semantic_body_part_clearance import semantic_body_part_clearance_reward

    body_pos = torch.zeros((1, 12, 3), dtype=torch.float32)
    body_pos[:, [0, 3, 6, 9], 2] = 0.30
    body_pos[:, [1, 4, 7, 10], 2] = 0.15
    body_pos[:, [2, 5, 8, 11], 2] = 0.00
    env = _FakeEnv()
    env.scene = _FakeScene(
        {
            "robot": _FakeRobot(body_pos),
            "semantic_height_scanner": _FakeScanner(),
            "contact_forces": _FakeContactSensor(),
        }
    )
    env.scene["semantic_height_scanner"].data.height_map[:, 1, 1] = 0.10
    env.scene["semantic_height_scanner"].data.semantic_map[:, 1, 1] = 1

    reward = semantic_body_part_clearance_reward(
        env,
        asset_cfg=_SceneEntity("robot"),
        scanner_cfg=_SceneEntity("semantic_height_scanner"),
        contact_sensor_cfg=_SceneEntity("contact_forces", body_names=".*_foot"),
    )

    assert reward.shape == (1,)
    assert torch.isfinite(reward).all()
    assert reward.item() < 0.0


def test_reward_wrapper_applies_clearance_scale() -> None:
    from extension.mdp.semantic_body_part_clearance import semantic_body_part_clearance_reward

    body_pos = torch.zeros((1, 12, 3), dtype=torch.float32)
    body_pos[:, [0, 3, 6, 9], 2] = 0.30
    body_pos[:, [1, 4, 7, 10], 2] = 0.15
    body_pos[:, [2, 5, 8, 11], 2] = 0.00
    env = _FakeEnv()
    env.scene = _FakeScene(
        {
            "robot": _FakeRobot(body_pos),
            "semantic_height_scanner": _FakeScanner(),
            "contact_forces": _FakeContactSensor(),
        }
    )
    env.scene["semantic_height_scanner"].data.height_map[:, 1, 1] = 0.10
    env.scene["semantic_height_scanner"].data.semantic_map[:, 1, 1] = 1

    raw = semantic_body_part_clearance_reward(
        env,
        asset_cfg=_SceneEntity("robot"),
        scanner_cfg=_SceneEntity("semantic_height_scanner"),
        contact_sensor_cfg=_SceneEntity("contact_forces", body_names=".*_foot"),
        clearance_scale=1.0,
    )
    scaled = semantic_body_part_clearance_reward(
        env,
        asset_cfg=_SceneEntity("robot"),
        scanner_cfg=_SceneEntity("semantic_height_scanner"),
        contact_sensor_cfg=_SceneEntity("contact_forces", body_names=".*_foot"),
        clearance_scale=1000.0,
    )

    assert raw.item() < 0.0
    assert scaled.item() == pytest.approx(raw.item() * 1000.0)
