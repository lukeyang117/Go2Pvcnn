from __future__ import annotations

import sys
import ast
from dataclasses import MISSING
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.batch_mpc_planner.config import MpcPlannerCfg, planner_cfg_from_task_cfg
from extension.batch_mpc_planner.kinematics import (
    fk_feet_from_joint_angles,
    fk_leg_points_from_joint_angles,
    solve_joint_angles_from_trajectory,
)
from extension.batch_mpc_planner.losses.contact import support_stability_loss
from extension.batch_mpc_planner.losses.gait_coupling import (
    diagonal_pair_loss,
    root_foot_center_loss,
    support_plane_roll_pitch_loss,
    swing_center_urgency_order_loss,
    swing_direction_loss,
)
from extension.batch_mpc_planner.losses.smoothness import (
    foot_acceleration_smoothness_loss,
    foot_boundary_smoothness_loss,
)
from extension.batch_mpc_planner.losses.terrain_clearance import (
    body_heightfield_collision_loss,
    high_obstacle_avoidance_loss,
    knee_shank_heightfield_collision_loss,
    high_large_stepcap_continuity_loss,
    low_small_foot_crossing_loss,
    low_small_foot_over_loss,
    low_small_stepcap_continuity_loss,
    low_small_crossing_progress_loss,
    obstacle_risk_scales,
    semantic_contact_avoidance_loss,
    semantic_obstacle_loss,
    stance_ground_loss,
    stance_semantic_obstacle_loss,
    swing_clearance_terrain_loss,
    touchdown_semantic_loss,
    touchdown_surface_loss,
)
from extension.batch_mpc_planner.losses.tracking import command_tracking_loss
from extension.batch_mpc_planner.manager import MpcTrajectoryManager
from extension.batch_mpc_planner.parametric_losses import (
    FkCollisionMargins,
    parametric_fk_body_leg_collision_loss,
    parametric_plane_root_z_target_loss,
    parametric_swing_foot_clearance_loss,
    parametric_touchdown_keepout_loss,
    parametric_trajectory_fk_consistency_loss,
)
from extension.batch_mpc_planner.parametric import decode_parametric_trajectory, init_parametric_variables
from extension.batch_mpc_planner.planner import (
    _command_farthest_touchdown_positions,
    _parametric_sampled_frame_losses,
    plan_segment,
    sample_touchdown_positions,
)
from extension.batch_mpc_planner.adapter import (
    clone_reference_cache,
    mpc_result_to_reference_cache,
    scatter_cache_rows,
)
from extension.batch_mpc_planner.semantic_policy import (
    SemanticObstacleMode,
    build_parametric_nominal,
    classify_semantic_obstacle_mode,
    shape_nominal_command_for_semantic_obstacles,
)
from extension.batch_mpc_planner.semantic_geometry import low_small_component_circles
from extension.batch_mpc_planner.terrain import (
    build_mpc_terrain_from_scanner,
    height_at,
    semantic_at,
    slope_at,
    subset_mpc_terrain,
    support_at,
)
from extension.batch_mpc_planner.losses.kinematics import ik_fk_residual_loss
from extension.batch_mpc_planner.losses.kinematics import ik_fk_residual_loss_from_joint_angles
from extension.batch_mpc_planner.losses.kinematics import joint_limit_loss_from_root_foot
from extension.batch_mpc_planner.types import MpcPlannerTerrain, MpcRobotState
from extension.mdp.observations import (
    _semantic_priority_pool2d,
    downsample_height_map,
    downsampled_elevation_semantic_scan,
)
from extension.mdp.rewards_reference import swing_leg_collision_reward
from mpc_low_small_reachable_crossing_probe import compute_plane_low_small_fk_metrics, compute_segmented_plane_low_small_fk_metrics
from extension.trajectory_manager_factory import create_trajectory_manager, planner_backend_from_cfg
from extension.viz.go2_foostep_planner import _adapt_mpc_result_for_viewer


PARAMETRIC_LOSS_KEYS = {
    "parametric_reachability",
    "parametric_terrain_clearance",
    "parametric_semantic_contact",
    "parametric_semantic_avoidance",
    "parametric_touchdown_keepout",
    "parametric_swing_foot_clearance",
    "parametric_fk_body_leg_collision",
    "parametric_trajectory_fk_consistency",
    "parametric_plane_root_z_target",
    "parametric_touchdown_endpoint",
    "parametric_foot_height_guard",
    "parametric_root_foot_center",
    "parametric_gait_regularization",
    "parametric_swing_direction",
    "parametric_command_progress",
    "parametric_curve_regularization",
    "parametric_joint_limit",
}


class DecodedTrajectoryStub(SimpleNamespace):
    def __init__(
        self,
        root_pos=None,
        root_rpy=None,
        foot_pos=None,
        swing_center=None,
        swing_width=None,
        swing_start=None,
        swing_end=None,
        swing_prob=None,
        contact_prob=None,
        **kwargs,
    ):
        super().__init__(
            root_pos=root_pos,
            root_rpy=root_rpy,
            foot_pos=foot_pos,
            swing_center=swing_center,
            swing_width=swing_width,
            swing_start=swing_start,
            swing_end=swing_end,
            swing_prob=swing_prob,
            contact_prob=contact_prob,
            **kwargs,
        )


def _task_cfg(**overrides):
    values = {
        "planner_backend": "mpc",
        "reference_command_name": "base_velocity",
        "reference_height_scanner_name": "height_scanner",
        "reference_trajectory_horizon": 6,
        "reference_replan_interval_steps": 3,
        "plan_dt": 0.02,
        "mpc_max_stale_steps": 6,
        "mpc_parallel_plan_batch_size": 2,
        "mpc_optimize_steps": 0,
        "mpc_diagnostics_enabled": False,
    }
    values.update(overrides)
    if "mpc_planner_cfg" not in values:
        values["mpc_planner_cfg"] = planner_cfg_from_task_cfg(SimpleNamespace(**values))
    return SimpleNamespace(**values)


def _make_simple_mpc_result(*, batch: int, horizon: int, offset: float = 0.0):
    root = torch.zeros((batch, horizon, 3), dtype=torch.float32)
    root[..., 0] = torch.arange(horizon, dtype=torch.float32).view(1, horizon) + float(offset)
    foot = root[:, :, None, :].expand(batch, horizon, 4, 3).clone()
    foot[..., 0] += torch.tensor([0.2, 0.2, -0.2, -0.2], dtype=torch.float32).view(1, 1, 4)
    foot[..., 1] += torch.tensor([0.1, -0.1, 0.1, -0.1], dtype=torch.float32).view(1, 1, 4)
    return SimpleNamespace(
        root_pos=root,
        root_rpy=torch.zeros((batch, horizon, 3), dtype=torch.float32),
        foot_pos=foot,
        joint_angles=torch.zeros((batch, horizon, 12), dtype=torch.float32),
        contact_state=torch.ones((batch, horizon, 4), dtype=torch.bool),
        planned_touchdown_w=foot,
    )


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor) -> None:
        self._command = command

    def get_command(self, name: str) -> torch.Tensor:
        assert name == "base_velocity"
        return self._command


class _FakeRobot:
    def __init__(self, *, num_envs: int, device: torch.device) -> None:
        root_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        root_pos[:, 2] = 0.30
        foot_offsets = torch.tensor(
            [
                [0.19, 0.05, -0.30],
                [0.19, -0.05, -0.30],
                [-0.19, 0.05, -0.30],
                [-0.19, -0.05, -0.30],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.data = SimpleNamespace(
            root_pos_w=root_pos,
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).expand(num_envs, -1),
            joint_pos=torch.zeros((num_envs, 12), dtype=torch.float32, device=device),
            body_pos_w=root_pos[:, None, :] + foot_offsets[None, :, :],
        )

    def find_bodies(self, _pattern: str):
        return [0, 1, 2, 3], ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class _FakeLegRobot:
    def __init__(self, body_pos_w: torch.Tensor) -> None:
        self.data = SimpleNamespace(
            body_pos_w=body_pos_w,
            body_names=[
                "FL_thigh",
                "FL_calf",
                "FL_foot",
                "FR_thigh",
                "FR_calf",
                "FR_foot",
            ],
        )

    def find_bodies(self, pattern):
        if pattern == ".*_foot":
            return [2, 5], ["FL_foot", "FR_foot"]
        return [0, 1, 2, 3, 4, 5], self.data.body_names


def test_swing_leg_collision_reward_uses_current_body_contacts_and_semantics() -> None:
    body_pos = torch.tensor(
        [
            [
                [-0.5, -0.5, 0.02],
                [-0.5, -0.5, 0.02],
                [-0.5, -0.5, 0.00],
                [0.5, 0.5, 0.02],
                [0.5, 0.5, 0.02],
                [0.5, 0.5, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    contact_forces = torch.zeros((1, 2, 3), dtype=torch.float32)
    contact_forces[0, 0, 2] = 5.0
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            elevation_map=torch.full((1, 3, 3), 0.05, dtype=torch.float32),
            semantic_map=torch.tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 2]]], dtype=torch.long),
            pos_w=torch.zeros((1, 3), dtype=torch.float32),
            quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
        cfg=SimpleNamespace(pattern_cfg=SimpleNamespace(size=(1.0, 1.0))),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        scene=SimpleNamespace(
            robot=_FakeLegRobot(body_pos),
            contact_forces=SimpleNamespace(data=SimpleNamespace(net_forces_w=contact_forces)),
            sensors=SimpleNamespace(
                semantic_height_scanner=scanner,
                contact_forces=SimpleNamespace(data=SimpleNamespace(net_forces_w=contact_forces)),
            ),
        ),
    )

    penalty = swing_leg_collision_reward(
        env,
        asset_cfg=SimpleNamespace(name="robot"),
        sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 1]),
        scanner_cfg=SimpleNamespace(name="semantic_height_scanner"),
        clearance=0.04,
        contact_force_threshold=1.0,
        stance_weight=0.25,
        swing_weight=1.0,
        small_obstacle_weight=2.0,
        large_obstacle_weight=5.0,
    )

    assert penalty.shape == (1,)
    assert penalty.item() < 0.0
    per_leg_clearance = 0.07 + 0.07 + 0.09
    stance_small = per_leg_clearance * 0.25 * 2.0
    swing_large = per_leg_clearance * 1.0 * 5.0
    assert penalty.item() == pytest.approx(-(stance_small + swing_large), rel=1.0e-5)


def test_downsampled_semantic_scan_preserves_priority_and_shape() -> None:
    elevation = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    semantic = torch.tensor(
        [
            [
                [0, 0, 1, 0],
                [0, 2, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 2],
            ]
        ],
        dtype=torch.long,
    )
    env = SimpleNamespace(
        num_envs=1,
        scene=SimpleNamespace(
            sensors=SimpleNamespace(
                semantic_height_scanner=SimpleNamespace(
                    data=SimpleNamespace(
                        elevation_map=elevation,
                        semantic_map=semantic,
                    )
                )
            )
        ),
    )

    obs = downsampled_elevation_semantic_scan(
        env,
        sensor_cfg=SimpleNamespace(name="semantic_height_scanner"),
        target_size=2,
    )

    assert obs.shape == (1, 2, 2, 2)
    torch.testing.assert_close(obs[:, 0], downsample_height_map(elevation, target_size=2))
    torch.testing.assert_close(obs[:, 1], torch.tensor([[[2.0, 1.0], [1.0, 2.0]]]))
    assert int(_semantic_priority_pool2d(semantic, target_size=1).item()) == 2


def _fake_env(*, num_envs: int = 3, device: torch.device | None = None, flatten_ray_hits: bool = False, terrain=None):
    device = device or torch.device("cpu")
    ray_hits_grid = torch.zeros((num_envs, 5, 5, 3), dtype=torch.float32, device=device)
    ray_hits_w = ray_hits_grid.reshape(num_envs, -1, 3) if flatten_ray_hits else ray_hits_grid
    semantic_map = torch.zeros((num_envs, 5, 5), dtype=torch.long, device=device)
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=ray_hits_w,
            semantic_map=semantic_map,
            pos_w=torch.zeros((num_envs, 3), dtype=torch.float32, device=device),
            quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).expand(num_envs, -1),
        ),
        cfg=SimpleNamespace(pattern_cfg=SimpleNamespace(size=(1.0, 1.0))),
    )
    commands = torch.tensor(
        [[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.0, 0.10, 0.20]],
        dtype=torch.float32,
        device=device,
    )[:num_envs]
    return SimpleNamespace(
        scene=SimpleNamespace(
            robot=_FakeRobot(num_envs=num_envs, device=device),
            sensors=SimpleNamespace(height_scanner=scanner),
            terrain=terrain,
        ),
        command_manager=_FakeCommandManager(commands),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long, device=device),
        common_step_counter=0,
        _trajectory_reference_cache=None,
    )


class _SubsetOnlyScanner:
    def __init__(self, *, num_envs: int, device: torch.device) -> None:
        self.update_calls: list[torch.Tensor] = []
        self._data = SimpleNamespace(
            ray_hits_w=torch.zeros((num_envs, 5, 5, 3), dtype=torch.float32, device=device),
            semantic_map=torch.zeros((num_envs, 5, 5), dtype=torch.long, device=device),
            pos_w=torch.zeros((num_envs, 3), dtype=torch.float32, device=device),
            quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).expand(num_envs, -1),
        )
        self.cfg = SimpleNamespace(pattern_cfg=SimpleNamespace(size=(1.0, 1.0)))

    @property
    def data(self):
        raise AssertionError("manager should refresh selected scanner env ids before reading scanner data")

    def update_env_ids(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long).cpu()
        self.update_calls.append(ids.clone())
        return self._data


def _mpc_plan_inputs(*, batch: int = 2, horizon: int = 6):
    root_pos = torch.zeros((batch, 3), dtype=torch.float32)
    root_pos[:, 2] = 0.30
    foot_pos = torch.tensor(
        [
            [0.19, 0.05, 0.0],
            [0.19, -0.05, 0.0],
            [-0.19, 0.05, 0.0],
            [-0.19, -0.05, 0.0],
        ],
        dtype=torch.float32,
    ).expand(batch, -1, -1)
    state = MpcRobotState(
        root_pos=root_pos,
        root_rpy=torch.zeros((batch, 3), dtype=torch.float32),
        foot_pos=foot_pos,
        joint_angles=torch.zeros((batch, 12), dtype=torch.float32),
    )
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((batch, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((batch, 5, 5), dtype=torch.long),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    command = torch.tensor([[0.0, 0.0, 0.0], [0.20, 0.0, 0.0]], dtype=torch.float32)[:batch]
    cfg = MpcPlannerCfg()
    cfg.runtime.horizon_steps = horizon
    cfg.runtime.optimize_steps = 0
    cfg.diagnostics.enabled = True
    return terrain, state, command, cfg


def _semantic_obstacle_inputs(*, obstacle_id: int = 1, obstacle_height: float = 0.12, command: torch.Tensor | None = None):
    terrain, state, _, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    # World ranges map x/y directly to columns/rows. This places the object
    # ahead of the root along +x and inside the command corridor.
    height[:, 4, 6] = float(obstacle_height)
    semantic[:, 4, 6] = int(obstacle_id)
    terrain = MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-0.8, 0.8),
        world_y_range=(-0.8, 0.8),
    )
    command = command if command is not None else torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32)
    state = MpcRobotState(
        root_pos=state.root_pos[:1],
        root_rpy=state.root_rpy[:1],
        foot_pos=state.foot_pos[:1],
        joint_angles=state.joint_angles[:1],
        foot_vel=state.foot_vel[:1] if state.foot_vel is not None else None,
    )
    return terrain, state, command, cfg


def _terrain_with_low_small_square() -> MpcPlannerTerrain:
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    semantic[0, 4:6, 4:6] = 1
    return MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )


def test_touchdown_keepout_only_triggers_when_touchdown_on_semantic() -> None:
    terrain = _terrain_with_low_small_square()
    touchdown = torch.tensor(
        [[[0.0, 0.0, 0.1], [0.5, 0.5, 0.0], [0.6, 0.5, 0.0], [0.7, 0.5, 0.0]]],
        dtype=torch.float32,
    )

    loss = parametric_touchdown_keepout_loss(
        terrain,
        touchdown,
        radius_extra_m=0.05,
        max_components=8,
    )

    assert loss.item() > 0.0


def test_touchdown_keepout_is_zero_for_nonsemantic_touchdowns() -> None:
    terrain = _terrain_with_low_small_square()
    touchdown = torch.tensor(
        [[[0.5, 0.5, 0.0], [0.6, 0.5, 0.0], [0.7, 0.5, 0.0], [0.8, 0.5, 0.0]]],
        dtype=torch.float32,
    )

    loss = parametric_touchdown_keepout_loss(
        terrain,
        touchdown,
        radius_extra_m=0.05,
        max_components=8,
    )

    assert loss.item() == pytest.approx(0.0)


def test_swing_target_clearance_penalizes_target_below_height_map() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.full((1, 5, 5), 0.10, dtype=torch.float32),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    foot = torch.zeros((1, 25, 4, 3), dtype=torch.float32)
    foot[..., 2] = 0.105
    swing_prob = torch.ones((1, 25, 4), dtype=torch.float32)

    loss = parametric_swing_foot_clearance_loss(
        terrain,
        foot,
        swing_prob,
        margin_m=0.02,
    )

    assert loss.item() > 0.0


def test_fk_leg_points_exposes_shank_pos_world_alias() -> None:
    root = torch.zeros((1, 25, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    joint = torch.zeros((1, 25, 12), dtype=torch.float32)

    points = fk_leg_points_from_joint_angles(root, rpy, joint, shank_sample_count=3)

    assert points.foot_pos_world.shape == (1, 25, 4, 3)
    assert points.knee_pos_world.shape == (1, 25, 4, 3)
    assert points.shank_pos_world.shape == (1, 25, 4, 3, 3)


def test_fk_body_leg_collision_penalizes_shank_below_terrain() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.full((1, 5, 5), 0.10, dtype=torch.float32),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    root_pos = torch.zeros((1, 25, 3), dtype=torch.float32)
    root_pos[..., 2] = 0.30
    points = fk_leg_points_from_joint_angles(
        root_pos,
        torch.zeros_like(root_pos),
        torch.zeros((1, 25, 12), dtype=torch.float32),
        shank_sample_count=3,
    )
    low_shank = points.shank_sample_world.clone()
    low_shank[..., 2] = 0.05
    points = type(points)(
        foot_pos_world=points.foot_pos_world,
        knee_pos_world=points.knee_pos_world,
        shank_sample_world=low_shank,
    )

    loss = parametric_fk_body_leg_collision_loss(
        terrain,
        root_pos,
        points,
        margins=FkCollisionMargins(
            foot=0.015,
            knee=0.01,
            shank=0.01,
            root=0.02,
            underbody=0.015,
        ),
        underbody_sample_count=5,
    )

    assert loss.item() > 0.0


def test_fk_body_leg_collision_keeps_sparse_foot_collision_salient_across_horizon() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.full((1, 5, 5), 0.10, dtype=torch.float32),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )

    def make_loss(horizon: int) -> torch.Tensor:
        root_pos = torch.zeros((1, horizon, 3), dtype=torch.float32)
        root_pos[..., 2] = 0.30
        foot = torch.zeros((1, horizon, 4, 3), dtype=torch.float32)
        foot[..., 2] = 0.30
        foot[:, 0, 0, 2] = 0.05
        points = SimpleNamespace(
            foot_pos_world=foot,
            knee_pos_world=foot + torch.tensor([0.0, 0.0, 0.25]),
            shank_sample_world=(foot + torch.tensor([0.0, 0.0, 0.20])).unsqueeze(-2),
        )
        return parametric_fk_body_leg_collision_loss(
            terrain,
            root_pos,
            points,
            margins=FkCollisionMargins(
                foot=0.015,
                knee=0.01,
                shank=0.01,
                root=0.02,
                underbody=0.015,
            ),
            underbody_sample_count=5,
        )

    short = make_loss(4)
    long = make_loss(40)

    assert float(long.item()) >= 0.5 * float(short.item())


def test_trajectory_consistency_penalizes_absolute_and_root_relative_error() -> None:
    root = torch.zeros((1, 25, 3), dtype=torch.float32)
    rpy = torch.zeros((1, 25, 3), dtype=torch.float32)
    target = torch.zeros((1, 25, 4, 3), dtype=torch.float32)
    fk = target.clone()
    fk[..., 0] += 0.10

    loss = parametric_trajectory_fk_consistency_loss(root, rpy, target, fk)

    assert loss.item() > 0.0


def test_plane_root_z_target_only_applies_to_plane_rows() -> None:
    root = torch.zeros((2, 25, 3), dtype=torch.float32)
    root[:, :, 2] = 0.40
    state_root = torch.zeros((2, 3), dtype=torch.float32)
    state_root[:, 2] = 0.32
    plane = torch.tensor([True, False])

    loss = parametric_plane_root_z_target_loss(
        root,
        state_root,
        plane,
        target_height_m=None,
    )

    assert loss[0].item() > 0.0
    assert loss[1].item() == pytest.approx(0.0)


def test_plane_low_small_metrics_count_semantic_collision_and_fk_error() -> None:
    terrain = _terrain_with_low_small_square()
    target = torch.zeros((1, 25, 4, 3), dtype=torch.float32)
    target[:, :, 0, 0] = torch.linspace(-0.2, 0.2, 25).view(1, 25)
    target[:, :, 0, 1] = 0.0
    target[:, :, 0, 2] = 0.20
    fk_points = fk_leg_points_from_joint_angles(
        torch.zeros((1, 25, 3), dtype=torch.float32),
        torch.zeros((1, 25, 3), dtype=torch.float32),
        torch.zeros((1, 25, 12), dtype=torch.float32),
        shank_sample_count=2,
    )
    low_foot = fk_points.foot_pos_world.clone()
    low_foot[:, :, 0, :2] = target[:, :, 0, :2]
    low_foot[:, :, 0, 2] = -0.01
    fk_points = type(fk_points)(
        foot_pos_world=low_foot,
        knee_pos_world=fk_points.knee_pos_world,
        shank_sample_world=fk_points.shank_sample_world,
    )

    metrics = compute_plane_low_small_fk_metrics(
        target_foot_pos=target,
        fk_points=fk_points,
        terrain=terrain,
        plane_mask=torch.tensor([True]),
        probe_half_width_m=0.06,
        probe_count=3,
    )

    assert "fk_semantic_collision_count" in metrics
    assert "planned_vs_fk_foot_error_crossing_leg_max_m" in metrics
    assert int(metrics["crossing_leg_count"]) >= 1
    assert int(metrics["fk_semantic_collision_count"]) > 0
    assert float(metrics["planned_vs_fk_foot_error_crossing_leg_max_m"]) > 0.0


def test_plane_low_small_metrics_ignore_non_crossing_leg_collision() -> None:
    terrain = _terrain_with_low_small_square()
    target = torch.zeros((1, 25, 4, 3), dtype=torch.float32)
    target[..., 0] = -0.35
    target[..., 1] = -0.35
    low_foot = target.clone()
    low_knee = torch.zeros_like(target)
    low_knee[:, :, 1, 0] = torch.linspace(-0.2, 0.2, 25).view(1, 25)
    low_knee[:, :, 1, 1] = 0.0
    low_knee[:, :, 1, 2] = -0.01
    fk_points = SimpleNamespace(
        foot_pos_world=low_foot,
        knee_pos_world=low_knee,
        shank_sample_world=torch.zeros((1, 25, 4, 2, 3), dtype=torch.float32),
    )

    metrics = compute_plane_low_small_fk_metrics(
        target_foot_pos=target,
        fk_points=fk_points,
        terrain=terrain,
        plane_mask=torch.tensor([True]),
        probe_half_width_m=0.06,
        probe_count=3,
    )

    assert int(metrics["crossing_leg_count"]) == 0
    assert int(metrics["fk_semantic_collision_count"]) == 0


def test_segmented_plane_low_small_metrics_uses_matching_segment_terrain() -> None:
    terrain_hit = _terrain_with_low_small_square()
    empty_semantic = torch.zeros_like(terrain_hit.semantic_map)
    terrain_clear = MpcPlannerTerrain(
        height_map=terrain_hit.height_map.clone(),
        semantic_map=empty_semantic,
        world_x_range=terrain_hit.world_x_range,
        world_y_range=terrain_hit.world_y_range,
        is_plane_terrain=torch.tensor([True]),
    )
    target = torch.zeros((1, 50, 4, 3), dtype=torch.float32)
    target[:, :25, :, 0] = -0.35
    target[:, :25, :, 1] = -0.35
    target[:, 25:, 0, 0] = torch.linspace(-0.2, 0.2, 25).view(1, 25)
    target[:, 25:, 0, 1] = 0.0
    target[:, :, 0, 2] = 0.20
    foot = target.clone()
    foot[:, 25:, 0, 2] = -0.01
    fk_points = SimpleNamespace(
        foot_pos_world=foot,
        knee_pos_world=torch.zeros_like(foot),
        shank_sample_world=torch.zeros((1, 50, 4, 2, 3), dtype=torch.float32),
    )

    stale = compute_plane_low_small_fk_metrics(
        target_foot_pos=target,
        fk_points=fk_points,
        terrain=terrain_hit,
        plane_mask=torch.tensor([True]),
        probe_half_width_m=0.06,
        probe_count=3,
    )
    segmented = compute_segmented_plane_low_small_fk_metrics(
        target_foot_pos=target,
        fk_points=fk_points,
        terrains=(terrain_hit, terrain_clear),
        segment_lengths=(25, 25),
        probe_half_width_m=0.06,
        probe_count=3,
    )

    assert int(stale["fk_semantic_collision_count"]) > 0
    assert int(segmented["fk_semantic_collision_count"]) == 0
    assert int(segmented["crossing_leg_count"]) == 0


def test_parametric_plan_exports_fk_realized_feet() -> None:
    terrain, state, _command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    cfg.runtime.optimize_steps = 0

    result = plan_segment(terrain, state, command, cfg=cfg)
    fk = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)

    torch.testing.assert_close(result.foot_pos[:, 1:], fk[:, 1:], atol=1.0e-5, rtol=1.0e-5)


def test_plan_segment_defaults_to_parametric_fk_realized_feet() -> None:
    terrain, state, _command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment(terrain, state, command, cfg=cfg)
    fk = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)

    torch.testing.assert_close(result.foot_pos[:, 1:], fk[:, 1:], atol=1.0e-5, rtol=1.0e-5)


def test_plan_segment_flat_body_forward_tracks_root_yaw_direction() -> None:
    terrain, state, _command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.optimize_steps = 0
    state = MpcRobotState(
        root_pos=state.root_pos,
        root_rpy=torch.tensor([[0.0, 0.0, torch.pi / 2.0]], dtype=torch.float32),
        foot_pos=state.foot_pos,
        joint_angles=state.joint_angles,
    )
    command = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment(terrain, state, command, cfg=cfg)

    delta_xy = result.root_pos[0, -1, :2] - state.root_pos[0, :2]
    direction = delta_xy / torch.linalg.vector_norm(delta_xy).clamp_min(1.0e-6)
    torch.testing.assert_close(direction, torch.tensor([0.0, 1.0], dtype=torch.float32), atol=1.0e-5, rtol=1.0e-5)
    assert abs(float(delta_xy[0])) <= 1.0e-5
    assert float(delta_xy[1]) > 0.05


def test_parametric_plan_exposes_sampled_frame_losses() -> None:
    terrain, state, _command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    cfg.diagnostics.enabled = True

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert PARAMETRIC_LOSS_KEYS.issubset(result.cost_breakdown)
    assert result.loss_breakdown is not None
    assert PARAMETRIC_LOSS_KEYS.issubset(result.loss_breakdown)
    for name in PARAMETRIC_LOSS_KEYS:
        assert result.cost_breakdown[name].shape == (1,)
        assert torch.isfinite(result.cost_breakdown[name]).all()


def test_parametric_sampled_losses_include_fk_optimization_terms() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)

    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=decoded.target_foot_pos,
        target_foot_pos=decoded.target_foot_pos,
        decoded=decoded,
        cfg=cfg,
    )

    assert "parametric_fk_body_leg_collision" in losses
    assert "parametric_trajectory_fk_consistency" in losses
    assert losses["parametric_fk_body_leg_collision"].shape == (1,)
    assert losses["parametric_trajectory_fk_consistency"].shape == (1,)


def test_parametric_trajectory_fk_consistency_uses_existing_ik_fk_weight() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    shifted_target = decoded.target_foot_pos.clone()
    shifted_target[..., 0] += 0.05

    cfg.losses.ik_fk_residual.weight = 3.0
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=decoded.target_foot_pos,
        target_foot_pos=shifted_target,
        decoded=decoded,
        cfg=cfg,
    )
    fk_joint = solve_joint_angles_from_trajectory(decoded.root_pos, decoded.root_rpy, shifted_target)
    fk_joint = fk_joint.clone()
    fk_joint[:, 0, :] = state.joint_angles
    fk_foot = fk_feet_from_joint_angles(decoded.root_pos, decoded.root_rpy, fk_joint)
    fk_foot = fk_foot.clone()
    fk_foot[:, 0, :, :] = state.foot_pos
    raw = parametric_trajectory_fk_consistency_loss(
        decoded.root_pos,
        decoded.root_rpy,
        shifted_target,
        fk_foot,
    )

    torch.testing.assert_close(losses["parametric_trajectory_fk_consistency"], 3.0 * raw)


def test_parametric_joint_limit_uses_existing_kinematics_weight_and_margin() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    shifted_target = decoded.target_foot_pos.clone()
    shifted_target[..., 1] += 0.20

    cfg.losses.kinematics.weight = 5.0
    cfg.losses.kinematics.joint_limit_margin_rad = 0.18
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=decoded.target_foot_pos,
        target_foot_pos=shifted_target,
        decoded=decoded,
        cfg=cfg,
    )
    raw = joint_limit_loss_from_root_foot(
        decoded.root_pos,
        decoded.root_rpy,
        shifted_target,
        joint_limit_margin_rad=0.18,
    )

    torch.testing.assert_close(losses["parametric_joint_limit"], 5.0 * raw)


def test_parametric_command_progress_uses_existing_progress_weight_and_min_progress() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    short_root = decoded.root_pos.clone()
    short_root[:, -1, 0] = 0.05

    cfg.losses.progress.weight = 7.0
    cfg.losses.progress.min_progress_m = 0.20
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=short_root,
        foot_pos=decoded.target_foot_pos,
        target_foot_pos=decoded.target_foot_pos,
        decoded=decoded,
        cfg=cfg,
    )

    expected = torch.tensor([(0.20 - 0.05) ** 2 * 7.0], dtype=torch.float32)
    torch.testing.assert_close(losses["parametric_command_progress"], expected)


def test_parametric_command_progress_penalizes_flat_empty_lateral_direction_error() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    terrain = MpcPlannerTerrain(
        height_map=terrain.height_map,
        semantic_map=torch.zeros_like(terrain.semantic_map),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        is_plane_terrain=torch.tensor([True]),
    )
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    lateral_root = decoded.root_pos.clone()
    lateral_root[:, -1, 0] = 0.20
    lateral_root[:, -1, 1] = 0.10

    cfg.losses.progress.weight = 4.0
    cfg.losses.progress.min_progress_m = 0.20
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=lateral_root,
        foot_pos=decoded.target_foot_pos,
        target_foot_pos=decoded.target_foot_pos,
        decoded=decoded,
        cfg=cfg,
    )

    expected = torch.tensor([0.10**2 * 4.0], dtype=torch.float32)
    torch.testing.assert_close(losses["parametric_command_progress"], expected)


def test_parametric_swing_direction_uses_existing_swing_direction_weight() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    terrain = MpcPlannerTerrain(
        height_map=terrain.height_map,
        semantic_map=torch.zeros_like(terrain.semantic_map),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        is_plane_terrain=torch.tensor([True]),
    )
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    lateral_foot = decoded.target_foot_pos.clone()
    lateral_foot[:, -1, :, 1] += 0.10

    cfg.losses.swing_direction.weight = 6.0
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=lateral_foot,
        target_foot_pos=lateral_foot,
        decoded=decoded,
        cfg=cfg,
    )

    assert "parametric_swing_direction" in losses
    assert float(losses["parametric_swing_direction"].item()) > 0.0


def test_parametric_swing_direction_uses_fk_realized_feet_for_direction_metric_alignment() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    terrain = MpcPlannerTerrain(
        height_map=terrain.height_map,
        semantic_map=torch.zeros_like(terrain.semantic_map),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        is_plane_terrain=torch.tensor([True]),
    )
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    target = decoded.target_foot_pos.clone()
    target[:, -1, :, 1] += 0.10
    fk_joint = solve_joint_angles_from_trajectory(decoded.root_pos, decoded.root_rpy, target)
    fk_foot = fk_feet_from_joint_angles(decoded.root_pos, decoded.root_rpy, fk_joint)
    fk_foot = fk_foot.clone()
    fk_foot[:, -1, :, 1] += 0.20

    cfg.losses.swing_direction.weight = 6.0
    target_losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=target,
        target_foot_pos=target,
        decoded=decoded,
        cfg=cfg,
    )
    fk_losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=fk_foot,
        target_foot_pos=target,
        decoded=decoded,
        cfg=cfg,
    )

    assert float(fk_losses["parametric_swing_direction"].item()) > float(
        target_losses["parametric_swing_direction"].item()
    )


def test_parametric_swing_direction_penalizes_whole_segment_fk_foot_lateral_drift() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    terrain = MpcPlannerTerrain(
        height_map=terrain.height_map,
        semantic_map=torch.zeros_like(terrain.semantic_map),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        is_plane_terrain=torch.tensor([True]),
    )
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    aligned = decoded.target_foot_pos.clone()
    aligned[:, -1, :, 0] += 0.20
    lateral = aligned.clone()
    lateral[:, -1, :, 1] += 0.20

    cfg.losses.swing_direction.weight = 6.0
    aligned_losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=aligned,
        target_foot_pos=aligned,
        decoded=decoded,
        cfg=cfg,
    )
    lateral_losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=lateral,
        target_foot_pos=lateral,
        decoded=decoded,
        cfg=cfg,
    )

    assert float(lateral_losses["parametric_swing_direction"].item()) > float(
        aligned_losses["parametric_swing_direction"].item()
    ) + 0.10


def test_swing_direction_loss_keeps_single_bad_leg_salient() -> None:
    horizon = 25
    root_pos = torch.zeros((1, horizon, 3), dtype=torch.float32)
    root_rpy = torch.zeros((1, horizon, 3), dtype=torch.float32)
    foot_pos = torch.zeros((1, horizon, 4, 3), dtype=torch.float32)
    foot_pos[:, :, :, 0] = torch.linspace(0.0, 0.20, horizon).view(1, horizon, 1)
    single_bad = foot_pos.clone()
    single_bad[:, :, 2, 1] = torch.linspace(0.0, 0.20, horizon).view(1, horizon)
    all_bad = foot_pos.clone()
    all_bad[:, :, :, 1] = torch.linspace(0.0, 0.20, horizon).view(1, horizon, 1)
    swing_center = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    swing_width = torch.full((1, 4), 1.0, dtype=torch.float32)
    command = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32)
    runtime = MpcPlannerCfg().runtime
    runtime.horizon_steps = horizon

    single = swing_direction_loss(root_pos, root_rpy, single_bad, swing_center, swing_width, command, runtime)
    all_legs = swing_direction_loss(root_pos, root_rpy, all_bad, swing_center, swing_width, command, runtime)

    assert float(single.item()) >= 0.5 * float(all_legs.item())


def test_parametric_swing_direction_does_not_constrain_semantic_obstacle_crossing() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    semantic = torch.zeros_like(terrain.semantic_map)
    semantic[:, 2, 2] = 1
    terrain = MpcPlannerTerrain(
        height_map=terrain.height_map,
        semantic_map=semantic,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        is_plane_terrain=torch.tensor([True]),
    )
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    lateral_foot = decoded.target_foot_pos.clone()
    lateral_foot[:, -1, :, 1] += 0.10

    cfg.losses.swing_direction.weight = 6.0
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=lateral_foot,
        target_foot_pos=lateral_foot,
        decoded=decoded,
        cfg=cfg,
    )

    torch.testing.assert_close(losses["parametric_swing_direction"], torch.zeros((1,), dtype=torch.float32))


def test_parametric_swing_direction_applies_when_flat_metadata_is_unavailable_but_semantics_are_empty() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    terrain = MpcPlannerTerrain(
        height_map=terrain.height_map,
        semantic_map=torch.zeros_like(terrain.semantic_map),
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
        is_plane_terrain=None,
    )
    nominal = build_parametric_nominal(state, terrain, command, cfg, horizon=25)
    variables = init_parametric_variables(state, nominal.command, horizon=25)
    decoded = decode_parametric_trajectory(state, terrain, nominal, variables, horizon=25)
    lateral_foot = decoded.target_foot_pos.clone()
    lateral_foot[:, -1, :, 1] += 0.10

    cfg.losses.swing_direction.weight = 6.0
    losses = _parametric_sampled_frame_losses(
        terrain,
        state,
        command,
        root_pos=decoded.root_pos,
        foot_pos=lateral_foot,
        target_foot_pos=lateral_foot,
        decoded=decoded,
        cfg=cfg,
    )

    assert float(losses["parametric_swing_direction"].item()) > 0.0


def test_parametric_optimization_exposes_touchdown_keepout_cost() -> None:
    terrain, state, _command, base_cfg = _semantic_obstacle_inputs(
        obstacle_id=1,
        obstacle_height=0.12,
        command=torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32),
    )
    no_opt_cfg = MpcPlannerCfg()
    no_opt_cfg.runtime.horizon_steps = 25
    no_opt_cfg.runtime.optimize_steps = 0
    no_opt_cfg.diagnostics.enabled = True
    opt_cfg = MpcPlannerCfg()
    opt_cfg.runtime.horizon_steps = 25
    opt_cfg.runtime.optimize_steps = 4
    opt_cfg.runtime.lr = 5.0e-2
    opt_cfg.diagnostics.enabled = True

    no_opt = plan_segment(terrain, state, _command, cfg=no_opt_cfg)
    optimized = plan_segment(terrain, state, _command, cfg=opt_cfg)

    assert optimized.loss_breakdown is not None
    assert "parametric_touchdown_keepout" in optimized.loss_breakdown
    assert optimized.cost_total.item() <= no_opt.cost_total.item()


def test_parametric_optimization_reduces_high_large_semantic_avoidance_cost() -> None:
    terrain, state, command, _cfg = _semantic_obstacle_inputs(
        obstacle_id=2,
        obstacle_height=0.45,
        command=torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32),
    )
    no_opt_cfg = MpcPlannerCfg()
    no_opt_cfg.runtime.horizon_steps = 25
    no_opt_cfg.runtime.optimize_steps = 0
    no_opt_cfg.diagnostics.enabled = True
    opt_cfg = MpcPlannerCfg()
    opt_cfg.runtime.horizon_steps = 25
    opt_cfg.runtime.optimize_steps = 6
    opt_cfg.runtime.lr = 5.0e-2
    opt_cfg.diagnostics.enabled = True

    no_opt = plan_segment(terrain, state, command, cfg=no_opt_cfg)
    optimized = plan_segment(terrain, state, command, cfg=opt_cfg)

    assert optimized.loss_breakdown is not None
    assert "parametric_semantic_avoidance" in optimized.loss_breakdown
    assert optimized.cost_total.item() <= no_opt.cost_total.item()


def test_parametric_plan_shapes_root_laterally_around_high_large_obstacle() -> None:
    terrain, state, command, _cfg = _semantic_obstacle_inputs(
        obstacle_id=2,
        obstacle_height=0.45,
        command=torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32),
    )
    cfg = MpcPlannerCfg()
    cfg.runtime.horizon_steps = 25
    cfg.runtime.optimize_steps = 0

    result = plan_segment(terrain, state, command, cfg=cfg)

    lateral = torch.abs(result.root_pos[0, :, 1]).amax().item()
    assert lateral > 0.05


def test_parametric_plan_keeps_high_large_touchdowns_off_semantic_cells() -> None:
    terrain, state, command, _cfg = _semantic_obstacle_inputs(
        obstacle_id=2,
        obstacle_height=0.45,
        command=torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32),
    )
    cfg = MpcPlannerCfg()
    cfg.runtime.horizon_steps = 25
    cfg.runtime.optimize_steps = 0

    result = plan_segment(terrain, state, command, cfg=cfg)
    touchdown_semantic = semantic_at(terrain, result.planned_touchdown_w[0, 0, :, :2].unsqueeze(0))

    assert torch.count_nonzero(touchdown_semantic).item() == 0


def test_parametric_plan_keeps_root_outside_large_obstacle_policy_margin() -> None:
    terrain, state, command, _cfg = _semantic_obstacle_inputs(
        obstacle_id=2,
        obstacle_height=0.45,
        command=torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32),
    )
    cfg = MpcPlannerCfg()
    cfg.runtime.horizon_steps = 25
    cfg.runtime.optimize_steps = 0

    result = plan_segment(terrain, state, command, cfg=cfg)
    obstacle_xy = torch.tensor([0.40, 0.0], dtype=torch.float32)
    min_distance = torch.linalg.vector_norm(result.root_pos[0, :, :2] - obstacle_xy, dim=-1).amin().item()

    assert min_distance >= 0.305


def test_parametric_optimization_keeps_pure_yaw_root_outside_large_obstacle_policy_margin() -> None:
    terrain, state, command, _cfg = _semantic_obstacle_inputs(
        obstacle_id=2,
        obstacle_height=0.45,
        command=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
    )
    cfg = MpcPlannerCfg()
    cfg.runtime.horizon_steps = 25
    cfg.runtime.optimize_steps = 6
    cfg.runtime.lr = 5.0e-2

    result = plan_segment(terrain, state, command, cfg=cfg)
    obstacle_xy = torch.tensor([0.40, 0.0], dtype=torch.float32)
    min_distance = torch.linalg.vector_norm(result.root_pos[0, :, :2] - obstacle_xy, dim=-1).amin().item()

    assert min_distance >= 0.305


def test_parametric_losses_include_endpoint_and_foot_height_guards() -> None:
    terrain, state, _command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.50, 0.25, 1.0]], dtype=torch.float32)
    cfg.runtime.optimize_steps = 0
    cfg.diagnostics.enabled = True

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert "parametric_touchdown_endpoint" in result.loss_breakdown
    assert "parametric_foot_height_guard" in result.loss_breakdown
    assert torch.isfinite(result.loss_breakdown["parametric_touchdown_endpoint"]).all()
    assert torch.isfinite(result.loss_breakdown["parametric_foot_height_guard"]).all()


def test_mpc_fk_leg_points_exposes_knee_and_shank_samples() -> None:
    root = torch.zeros((2, 3, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    joint = torch.zeros((2, 3, 12), dtype=torch.float32)

    leg_points = fk_leg_points_from_joint_angles(root, rpy, joint, shank_sample_count=2)

    assert leg_points.foot_pos_world.shape == (2, 3, 4, 3)
    assert leg_points.knee_pos_world.shape == (2, 3, 4, 3)
    assert leg_points.shank_sample_world.shape == (2, 3, 4, 2, 3)
    torch.testing.assert_close(
        leg_points.foot_pos_world,
        fk_feet_from_joint_angles(root, rpy, joint),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.isfinite(leg_points.knee_pos_world).all()
    assert torch.isfinite(leg_points.shank_sample_world).all()


def test_build_mpc_terrain_accepts_flattened_ray_hits_and_subset_batch_dimension() -> None:
    ray_hits = torch.zeros((4, 25, 3), dtype=torch.float32)
    semantic_map = torch.zeros((4, 25), dtype=torch.long)
    terrain = build_mpc_terrain_from_scanner(
        ray_hits,
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
        semantic_map=semantic_map,
    )

    assert terrain.height_map.shape == (4, 5, 5)
    assert terrain.semantic_map is not None
    assert terrain.semantic_map.shape == (4, 5, 5)

    sub = subset_mpc_terrain(terrain, torch.tensor([1, 3], dtype=torch.long))
    assert sub.height_map.shape == (2, 5, 5)
    assert sub.semantic_map is not None
    assert sub.semantic_map.shape == (2, 5, 5)


def test_mpc_terrain_preserves_is_plane_terrain_metadata() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((2, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((2, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
        is_plane_terrain=torch.tensor([True, False]),
    )

    sub = subset_mpc_terrain(terrain, torch.tensor([1], dtype=torch.long))

    assert sub.is_plane_terrain is not None
    assert sub.is_plane_terrain.tolist() == [False]


def test_build_mpc_terrain_from_scanner_carries_is_plane_terrain_metadata() -> None:
    ray_hits = torch.zeros((2, 25, 3), dtype=torch.float32)
    is_plane = torch.tensor([True, False])

    terrain = build_mpc_terrain_from_scanner(
        ray_hits,
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
        is_plane_terrain=is_plane,
    )

    assert terrain.is_plane_terrain is not None
    assert terrain.is_plane_terrain.dtype == torch.bool
    assert terrain.is_plane_terrain.tolist() == [True, False]


def test_low_small_gpu_circles_split_disconnected_components() -> None:
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    semantic[0, 2:4, 2:4] = 1
    semantic[0, 6:8, 6:8] = 1

    circles = low_small_component_circles(
        semantic,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
        max_components=4,
    )

    assert circles.center_xy.shape == (1, 4, 2)
    assert circles.radius.shape == (1, 4)
    assert circles.valid.shape == (1, 4)
    assert int(circles.valid[0].sum()) == 2
    assert circles.truncated.tolist() == [False]
    assert torch.all(circles.radius[circles.valid] > 0.0)


def test_low_small_gpu_circles_stay_on_input_device() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    semantic = torch.zeros((1, 9, 9), dtype=torch.long, device=device)
    semantic[0, 3:6, 3:6] = 1

    circles = low_small_component_circles(
        semantic,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
        max_components=4,
    )

    assert circles.center_xy.device == semantic.device
    assert circles.radius.device == semantic.device
    assert circles.valid.device == semantic.device


def test_mpc_terrain_height_semantic_slope_and_support_queries() -> None:
    height = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.2],
                [0.0, 0.2, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    semantic = torch.tensor(
        [
            [
                [0, 0, 0],
                [0, 1, 2],
                [0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )
    terrain = MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    query = torch.tensor([[[0.0, 0.0], [0.9, 0.0]]], dtype=torch.float32)

    sampled_h = height_at(terrain, query)
    sampled_sem = semantic_at(terrain, query)
    sampled_slope = slope_at(terrain, query, sample_step=0.25)
    support_xy, support_z, support_slope, invalid = support_at(
        terrain,
        query,
        search_radius=0.5,
        search_step=0.25,
        max_support_slope=1.0,
    )

    assert sampled_h.shape == (1, 2)
    assert sampled_sem.shape == (1, 2)
    assert sampled_slope.shape == (1, 2)
    assert support_xy.shape == (1, 2, 2)
    assert support_z.shape == (1, 2)
    assert support_slope.shape == (1, 2)
    assert invalid.shape == (1, 2)
    assert sampled_sem[0, 0].item() == 1
    assert not bool(invalid[0, 0].item())


def test_mpc_touchdown_surface_loss_has_finite_flat_ground_gradients() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    touchdown = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.2, 0.1, 0.0], [-0.2, 0.1, 0.0], [0.0, -0.2, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss = touchdown_surface_loss(
        terrain,
        touchdown,
        slope_sample_step=0.05,
        support_search_radius=0.10,
        support_search_step=0.05,
        max_slope=0.6,
        max_support_slope=0.6,
        support_height_tolerance=0.03,
        ground_weight=1.0,
        slope_weight=1.0,
        support_distance_weight=1.0,
        support_height_weight=1.0,
        support_slope_weight=1.0,
        invalid_support_weight=1.0,
    ).mean()
    loss.backward()

    assert touchdown.grad is not None
    assert torch.isfinite(touchdown.grad).all()


def test_mpc_terrain_queries_use_per_env_scanner_pose_for_world_points() -> None:
    ray_hits = torch.zeros((2, 3, 3, 3), dtype=torch.float32)
    ray_hits[0, 1, 1, 2] = 1.25
    ray_hits[1, 1, 1, 2] = 2.50
    terrain = build_mpc_terrain_from_scanner(
        ray_hits,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
        sensor_pos_w=torch.tensor([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]], dtype=torch.float32),
        sensor_yaw=torch.zeros(2, dtype=torch.float32),
    )

    query = torch.tensor([[[10.0, 0.0]], [[-10.0, 0.0]]], dtype=torch.float32)
    sampled = height_at(terrain, query)

    torch.testing.assert_close(sampled[:, 0], torch.tensor([1.25, 2.50], dtype=torch.float32))


def test_mpc_terrain_queries_preserve_scanner_local_positive_y() -> None:
    ray_hits = torch.zeros((1, 3, 3, 3), dtype=torch.float32)
    ray_hits[0, 0, 1, 2] = 0.25
    ray_hits[0, 2, 1, 2] = 1.25
    semantic = torch.zeros((1, 3, 3), dtype=torch.long)
    semantic[0, 2, 1] = 1
    terrain = build_mpc_terrain_from_scanner(
        ray_hits,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
        semantic_map=semantic,
        sensor_pos_w=torch.zeros((1, 3), dtype=torch.float32),
        sensor_yaw=torch.zeros(1, dtype=torch.float32),
    )

    query = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)

    torch.testing.assert_close(height_at(terrain, query), torch.tensor([[1.25]], dtype=torch.float32))
    torch.testing.assert_close(semantic_at(terrain, query), torch.tensor([[1]], dtype=torch.long))


def test_mpc_manager_terrain_from_env_carries_scanner_pose() -> None:
    cfg = _task_cfg()
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=2)
    env.scene.sensors.height_scanner.data.ray_hits_w[:, 2, 2, 2] = torch.tensor([0.4, 0.8])
    env.scene.sensors.height_scanner.data.pos_w = torch.tensor([[3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=torch.float32)

    terrain = manager._terrain_from_env(env)
    sampled = height_at(terrain, torch.tensor([[[3.0, 0.0]], [[-2.0, 0.0]]], dtype=torch.float32))

    torch.testing.assert_close(sampled[:, 0], torch.tensor([0.4, 0.8], dtype=torch.float32))


@pytest.mark.parametrize("backend_name", ["mpc", "MPC"])
def test_factory_recognizes_mpc_backend(backend_name: str) -> None:
    cfg = _task_cfg(planner_backend=backend_name)

    assert planner_backend_from_cfg(cfg) == "mpc"
    manager = create_trajectory_manager(cfg, device="cpu")

    assert isinstance(manager, MpcTrajectoryManager)
    assert manager.planner_backend == "mpc"
    assert manager.horizon_steps() == cfg.mpc_planner_cfg.runtime.horizon_steps


def test_factory_rejects_unknown_backend_with_valid_backend_hint() -> None:
    with pytest.raises(ValueError, match="mpc"):
        planner_backend_from_cfg(_task_cfg(planner_backend="dense_mpc"))


def test_mpc_semantic_trajectory_cfg_defaults_to_mpc_and_semantic_scanner() -> None:
    cfg_path = GO2PVCNN_ROOT / "go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py"
    source = cfg_path.read_text()
    tree = ast.parse(source)

    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg" in class_names
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY" in class_names
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER" in class_names
    assert 'planner_backend: str = "mpc"' in source
    assert 'reference_height_scanner_name: str = "semantic_height_scanner"' in source
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    class_sources = {name: ast.get_source_segment(source, node) or "" for name, node in classes.items()}
    assert "planner_owned_reference_cache: bool = True" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg"]
    assert "use_batched_reference_trajectory: bool = True" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg"]
    assert "planner_owned_reference_cache: bool = False" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"]
    assert "use_batched_reference_trajectory: bool = False" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"]
    assert "self.rewards.reference_foot_pos = None" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"]
    assert "self.rewards.semantic_contact_collision = None" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"]
    assert "self.scene.semantic_contact_small = None" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"]
    assert "self.scene.semantic_contact_large = None" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"]
    assert "planner_owned_reference_cache: bool = True" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER"]
    assert "use_batched_reference_trajectory: bool = True" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER"]
    assert "self.rewards.semantic_contact_collision = _semantic_contact_collision_reward_term()" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER"]
    assert "self.rewards.reference_foot_pos = _reference_foot_pos_reward_term()" in class_sources["TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER"]
    assert "self.mpc_planner_cfg.runtime.parallel_plan_batch_size = 4096" in source
    assert "mpc_max_dirty_envs_per_step" not in source
    assert "replicate_physics = True" in source
    assert "replicate_physics=True" in source
    assert "SemanticCourseTerrainImporter" in source
    assert "self.scene.terrain.class_type = SemanticCourseTerrainImporter" in source
    assert "generate_semantic_course" not in source
    assert "teacher_elevation_trajectory_env_cfg" not in source
    assert "teacher_without_semantic_env_cfg" not in source
    assert "teacher_semantic_env_cfg" not in source
    assert "height_scanner = None" in source
    assert "semantic_height_scanner = SemanticGridRayCasterCfg" in source
    assert "resolution=0.01" in source
    assert "size=[1.5, 1.5]" in source
    assert "downsampled_elevation_semantic_scan" in source
    assert "reference_foot_pos" in source
    assert "semantic_contact_collision" in source
    assert "reference_root_pose" not in source
    assert "reference_joint_pos" not in source
    assert "reference_contact" not in source
    assert "reference_touchdown" not in source


def test_semantic_raycaster_refreshes_mesh_after_startup_course_generation() -> None:
    source = (GO2PVCNN_ROOT / "go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py").read_text()

    assert "_refresh_late_semantic_mesh_if_needed" in source
    assert "semantic ids 1/2" in source
    assert "self._refresh_late_semantic_mesh_if_needed()" in source
    assert "torch.unique(self._face_semantic_ids)" in source
    assert "late semantic root" in source
    assert "except RuntimeError as exc" in source


def test_mpc_semantic_train_play_parsers_and_gym_registration_are_isolated() -> None:
    train_source = (GO2PVCNN_ROOT / "scripts/train.py").read_text()
    play_source = (GO2PVCNN_ROOT / "scripts/play.py").read_text()
    register_source = (GO2PVCNN_ROOT / "go2_pvcnn/tasks/register_envs.py").read_text()
    agent_source = (GO2PVCNN_ROOT / "agent/train_cfg.py").read_text()
    factory_source = (GO2PVCNN_ROOT / "extension/trajectory_manager_factory.py").read_text()

    assert "teacher_elevation_trajectory_mpc_semantic" in train_source
    assert "teacher_elevation_trajectory_mpc_semantic" in play_source
    assert '"mpc"' in play_source
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg" in train_source
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY" in play_source
    assert "TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER" not in play_source
    assert "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0" in register_source
    assert "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0" in register_source
    assert "_teacher_elevation_trajectory_mpc_semantic_train_cfg" in agent_source
    assert "teacher_elevation_trajectory_mpc_semantic" in factory_source
    assert "env_cfg.use_batched_reference_trajectory = True" not in play_source


def test_cleanup_entrypoints_only_expose_mpc_semantic_experiment() -> None:
    train_source = (GO2PVCNN_ROOT / "scripts/train.py").read_text()
    play_source = (GO2PVCNN_ROOT / "scripts/play.py").read_text()
    register_source = (GO2PVCNN_ROOT / "go2_pvcnn/tasks/register_envs.py").read_text()

    assert "teacher_elevation_trajectory_mpc_semantic" in train_source
    assert "teacher_elevation_trajectory_mpc_semantic" in play_source
    assert "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0" in register_source
    forbidden = (
        "teacher_without_semantic",
        "teacher_semantic",
        "teacher_elevation_semantic_map",
        "teacher_elevation_trajectory\"",
        "Isaac-Teacher-Without-Semantic-Go2-v0",
        "Isaac-Teacher-Semantic-Go2-v0",
        "Isaac-Teacher-Elevation-Semantic-Map-Go2-v0",
        "Isaac-Teacher-Elevation-Trajectory-Go2-v0",
    )
    combined = "\n".join((train_source, play_source, register_source))
    for token in forbidden:
        assert token not in combined


def test_cleanup_mpc_factory_has_no_legacy_or_together_backend() -> None:
    source = (GO2PVCNN_ROOT / "extension/trajectory_manager_factory.py").read_text()

    assert "MpcTrajectoryManager" in source
    assert "batched_together_planner" not in source
    assert "batched_planner" not in source
    assert '"legacy"' not in source
    assert '"together"' not in source


def test_cleanup_batch_mpc_planner_has_no_debug_variants_module() -> None:
    root = GO2PVCNN_ROOT / "extension" / "batch_mpc_planner"
    assert not (root / "debug_variants.py").exists()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    assert "debug_loss_variant" not in combined
    assert "apply_mpc_debug_variant_cfg" not in combined


def test_mpc_manager_refreshes_reference_cache_and_returns_current_reference_shapes() -> None:
    cfg = _task_cfg()
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    cache = manager.refresh_from_env(env)

    assert cache is env._trajectory_reference_cache
    assert cache.is_ready(), cache.shape_issues()
    assert cache.root_pos_w.shape == (3, 6, 3)
    assert cache.root_quat_w.shape == (3, 6, 4)
    assert cache.joint_angles.shape == (3, 6, 12)
    assert cache.foot_pos_w.shape == (3, 6, 4, 3)
    assert cache.foot_pos_root.shape == (3, 6, 4, 3)
    assert cache.contact_state.shape == (3, 6, 4)
    assert cache.contact_state.dtype == torch.bool
    assert cache.planned_touchdown_w.shape == (3, 6, 4, 3)
    assert cache.phase_index.shape == (3, 6)
    assert cache.valid_mask.shape == (3, 6)

    current = manager.current_reference()

    assert set(current) == {
        "root_pos_w",
        "root_quat_w",
        "joint_angles",
        "foot_pos_w",
        "foot_pos_root",
        "contact_state",
        "planned_touchdown_w",
        "phase_index",
        "valid_mask",
    }
    assert current["root_pos_w"].shape == (3, 3)
    assert current["root_quat_w"].shape == (3, 4)
    assert current["joint_angles"].shape == (3, 12)
    assert current["foot_pos_w"].shape == (3, 4, 3)
    assert current["foot_pos_root"].shape == (3, 4, 3)
    assert current["contact_state"].shape == (3, 4)
    assert current["planned_touchdown_w"].shape == (3, 4, 3)
    assert current["phase_index"].shape == (3,)
    assert current["valid_mask"].shape == (3,)
    assert manager.current_frame_ids().shape == (3,)

    same_step_cache = manager.refresh_from_env(env)

    assert same_step_cache is cache
    assert env._trajectory_reference_cache is cache

    env.common_step_counter = 1
    next_step_cache = manager.refresh_from_env(env)

    assert next_step_cache is env._trajectory_reference_cache
    assert next_step_cache.root_pos_w.shape == (3, 6, 3)
    assert manager.current_frame_ids().shape == (3,)


def test_mpc_reference_cache_exports_world_feet() -> None:
    result = _make_simple_mpc_result(batch=2, horizon=5)
    cache = mpc_result_to_reference_cache(result)
    assert cache.foot_pos_w is not None
    assert cache.foot_pos_w.shape == (2, 5, 4, 3)
    torch.testing.assert_close(cache.foot_pos_w, result.foot_pos)


def test_reference_cache_clone_scatter_preserve_world_feet() -> None:
    result = _make_simple_mpc_result(batch=3, horizon=6)
    cache = mpc_result_to_reference_cache(result)
    cloned = clone_reference_cache(cache)
    assert cloned.foot_pos_w is not None
    assert cache.foot_pos_w is not None
    torch.testing.assert_close(cloned.foot_pos_w, cache.foot_pos_w)

    new = mpc_result_to_reference_cache(_make_simple_mpc_result(batch=1, horizon=6, offset=10.0))
    scatter_cache_rows(cloned, new, torch.tensor([1], device=cache.root_pos_w.device))
    assert new.foot_pos_w is not None
    torch.testing.assert_close(cloned.foot_pos_w[1], new.foot_pos_w[0])


def test_mpc_manager_global_sync_samples_parallel_plan_batch_size() -> None:
    device = torch.device("cpu")
    num_envs = 3
    scanner = _SubsetOnlyScanner(num_envs=num_envs, device=device)
    env = _fake_env(num_envs=num_envs, device=device)
    env.scene.sensors.height_scanner = scanner
    cfg = _task_cfg(mpc_parallel_plan_batch_size=2)
    manager = create_trajectory_manager(cfg, device="cpu")

    manager.refresh_from_env(env)

    assert scanner.update_calls
    assert int(scanner.update_calls[0].numel()) == 2
    assert all(int(call.numel()) <= 2 for call in scanner.update_calls)


def test_mpc_global_sync_reference_reward_mask_only_enables_sampled_rows() -> None:
    cfg = _task_cfg(mpc_parallel_plan_batch_size=2)
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    manager.refresh_from_env(env)

    mask = manager.reference_reward_mask()
    assert mask.shape == (3,)
    assert int(torch.count_nonzero(mask).item()) == 2
    assert mask.dtype == torch.bool


def test_mpc_global_sync_reset_or_command_change_disables_existing_plan_reward() -> None:
    cfg = _task_cfg(mpc_parallel_plan_batch_size=3)
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    manager.refresh_from_env(env)
    assert torch.all(manager.reference_reward_mask())

    reset_mask = torch.tensor([False, True, False])
    manager.reset_envs(reset_mask)
    mask_after_reset = manager.reference_reward_mask()
    assert mask_after_reset.tolist() == [True, False, True]

    command_mask = torch.tensor([True, False, False])
    manager.mark_command_changed(command_mask)
    mask_after_command = manager.reference_reward_mask()
    assert mask_after_command.tolist() == [False, False, True]


def test_mpc_global_sync_does_not_replan_unsampled_or_command_changed_rows_before_interval() -> None:
    cfg = _task_cfg(mpc_parallel_plan_batch_size=2, reference_replan_interval_steps=3)
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    manager.refresh_from_env(env)
    first_mask = manager.reference_reward_mask().clone()
    assert int(torch.count_nonzero(first_mask).item()) == 2

    manager.mark_command_changed(first_mask)
    env.common_step_counter = 1
    manager.refresh_from_env(env)
    second_mask = manager.reference_reward_mask()

    assert not bool(torch.any(second_mask).item())


def test_mpc_global_sync_keeps_existing_reward_mask_between_replan_ticks() -> None:
    cfg = _task_cfg(mpc_parallel_plan_batch_size=3, reference_replan_interval_steps=3)
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    manager.refresh_from_env(env)
    first_mask = manager.reference_reward_mask().clone()
    assert torch.all(first_mask)

    env.common_step_counter = 1
    manager.refresh_from_env(env)
    second_mask = manager.reference_reward_mask()

    assert second_mask.tolist() == first_mask.tolist()


def test_mpc_global_sync_safe_fallback_rows_do_not_enable_imitation_reward(monkeypatch) -> None:
    cfg = _task_cfg(mpc_parallel_plan_batch_size=2)
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    original_plan_segment = plan_segment

    def _fake_plan_segment(*args, **kwargs):
        result = original_plan_segment(*args, **kwargs)
        batch = int(result.root_pos.shape[0])
        feasible = torch.ones(batch, dtype=torch.bool, device=result.root_pos.device)
        safe_fallback = torch.zeros(batch, dtype=torch.bool, device=result.root_pos.device)
        feasible[0] = False
        safe_fallback[0] = True
        return type(result)(
            root_pos=result.root_pos,
            root_rpy=result.root_rpy,
            foot_pos=result.foot_pos,
            joint_angles=result.joint_angles,
            contact_state=result.contact_state,
            touchdown_seq=result.touchdown_seq,
            planned_touchdown_w=result.planned_touchdown_w,
            cost_total=result.cost_total,
            cost_breakdown=result.cost_breakdown,
            status=result.status,
            feasible=feasible,
            safe_fallback=safe_fallback,
            loss_breakdown=result.loss_breakdown,
            hard_reason_mask=result.hard_reason_mask,
        )

    monkeypatch.setattr("extension.batch_mpc_planner.manager.plan_segment", _fake_plan_segment)

    manager.refresh_from_env(env)
    mask = manager.reference_reward_mask()

    assert mask.shape == (3,)
    assert int(torch.count_nonzero(mask).item()) == 1
    assert not bool(mask[0].item())


def test_mpc_global_sync_nonfinite_result_rows_do_not_enable_imitation_reward(monkeypatch) -> None:
    cfg = _task_cfg(mpc_parallel_plan_batch_size=2)
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    original_plan_segment = plan_segment

    def _fake_plan_segment(*args, **kwargs):
        result = original_plan_segment(*args, **kwargs)
        root_pos = result.root_pos.clone()
        root_pos[0, 0, 0] = torch.nan
        return type(result)(
            root_pos=root_pos,
            root_rpy=result.root_rpy,
            foot_pos=result.foot_pos,
            joint_angles=result.joint_angles,
            contact_state=result.contact_state,
            touchdown_seq=result.touchdown_seq,
            planned_touchdown_w=result.planned_touchdown_w,
            cost_total=result.cost_total,
            cost_breakdown=result.cost_breakdown,
            status=result.status,
            feasible=result.feasible,
            safe_fallback=result.safe_fallback,
            loss_breakdown=result.loss_breakdown,
            hard_reason_mask=result.hard_reason_mask,
        )

    monkeypatch.setattr("extension.batch_mpc_planner.manager.plan_segment", _fake_plan_segment)

    manager.refresh_from_env(env)
    mask = manager.reference_reward_mask()

    assert mask.shape == (3,)
    assert int(torch.count_nonzero(mask).item()) == 1
    assert not bool(mask[0].item())


def test_mpc_manager_supports_flattened_scanner_ray_hits_shape() -> None:
    cfg = _task_cfg()
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3, flatten_ray_hits=True)

    cache = manager.refresh_from_env(env)

    assert cache.is_ready()
    assert cache.root_pos_w.shape == (3, 6, 3)


def test_mpc_manager_carries_plane_terrain_metadata_from_terrain_types() -> None:
    cfg = _task_cfg()
    manager = create_trajectory_manager(cfg, device="cpu")
    terrain_importer = SimpleNamespace(
        terrain_types=torch.tensor([0, 1, 0], dtype=torch.long),
        cfg=SimpleNamespace(
            terrain_generator=SimpleNamespace(
                sub_terrains={
                    "flat": SimpleNamespace(),
                    "stairs": SimpleNamespace(),
                }
            )
        ),
    )
    env = _fake_env(num_envs=3, terrain=terrain_importer)

    terrain = manager._terrain_from_env(env)
    sub = manager._terrain_subset_from_env(env, torch.tensor([1, 2], dtype=torch.long))

    assert terrain.is_plane_terrain is not None
    assert terrain.is_plane_terrain.tolist() == [True, False, True]
    assert sub.is_plane_terrain is not None
    assert sub.is_plane_terrain.tolist() == [False, True]


def test_mpc_result_and_package_do_not_depend_on_old_mode_fields() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs()

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.root_pos.shape == (2, 6, 3)
    assert result.foot_pos.shape == (2, 6, 4, 3)
    assert result.contact_state.shape == (2, 6, 4)
    assert result.touchdown_seq.shape == (2, 4, 2, 3)
    assert result.hard_reason_mask is not None
    assert result.hard_reason_mask.shape[0] == 2
    assert result.loss_breakdown is not None
    assert PARAMETRIC_LOSS_KEYS.issubset(result.loss_breakdown)
    forbidden_result_fields = (
        "mode",
        "state_mode",
        "small_strategy_outcome",
        "selected_beta",
        "selected_route",
        "semantic_candidate_costs",
        "candidate_hard_reason_mask",
        "selected_candidate_index",
    )
    for field_name in forbidden_result_fields:
        assert not hasattr(result, field_name), field_name

    forbidden_source_tokens = (
        "T116_MODE_",
        "TogetherPlanner",
        "TogetherRobotState",
        "batched_together_planner",
        "state_mode",
        "small_strategy_outcome",
        "selected_beta",
        "selected_route",
        "semantic_candidate_costs",
        "candidate_hard_reason_mask",
        "selected_candidate_index",
    )
    violations: list[str] = []
    for path in sorted((GO2PVCNN_ROOT / "extension" / "batch_mpc_planner").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden_source_tokens:
            if token in text:
                violations.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {token}")

    assert violations == []


def test_mpc_profile_prints_plan_optimizer_and_loss_stages(monkeypatch, capsys) -> None:
    terrain, state, command, cfg = _mpc_plan_inputs()
    cfg.runtime.optimize_steps = 1
    cfg.diagnostics.emit_runtime_counters = True
    cfg.diagnostics.profile_cuda_sync = False
    monkeypatch.setenv("T302G_MPC_PROFILE_LIMIT", "1")

    plan_segment(terrain, state, command, cfg=cfg)

    out = capsys.readouterr().out
    assert "[MPC profile]" in out
    assert "plan.normalize_ms=" in out
    assert "plan.parametric_ms=" in out
    assert "loss.total_ms=" in out












def test_mpc_debug_v12_touchdown_export_uses_command_farthest_swing_point() -> None:
    terrain, _, command, _ = _mpc_plan_inputs(batch=1, horizon=5)
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    foot_pos = torch.tensor(
        [
            [
                [[0.10, 0.05, 0.00], [0.19, -0.05, 0.00], [-0.19, 0.05, 0.00], [-0.19, -0.05, 0.00]],
                [[0.20, 0.05, 0.04], [0.19, -0.05, 0.00], [-0.19, 0.05, 0.00], [-0.19, -0.05, 0.00]],
                [[0.48, 0.05, 0.12], [0.19, -0.05, 0.00], [-0.19, 0.05, 0.00], [-0.19, -0.05, 0.00]],
                [[0.30, 0.05, 0.03], [0.19, -0.05, 0.00], [-0.19, 0.05, 0.00], [-0.19, -0.05, 0.00]],
                [[0.12, 0.05, 0.00], [0.19, -0.05, 0.00], [-0.19, 0.05, 0.00], [-0.19, -0.05, 0.00]],
            ]
        ],
        dtype=torch.float32,
    )
    contact_state = torch.ones((1, 5, 4), dtype=torch.bool)
    contact_state[:, :4, 0] = False

    touchdown = _command_farthest_touchdown_positions(terrain, foot_pos, contact_state, command)

    assert touchdown[0, 0, 0].item() == pytest.approx(0.48)
    assert touchdown[0, 0, 1].item() == pytest.approx(0.05)
    assert touchdown[0, 0, 2].item() == pytest.approx(0.0)






def test_mpc_plan_segment_keeps_frame_zero_joint_state_anchor() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=2, horizon=25)
    cfg.runtime.optimize_steps = 1
    state = MpcRobotState(
        root_pos=state.root_pos,
        root_rpy=state.root_rpy,
        foot_pos=state.foot_pos,
        joint_angles=torch.arange(24, dtype=torch.float32).reshape(2, 12) * 0.01,
    )

    result = plan_segment(terrain, state, command, cfg=cfg)

    torch.testing.assert_close(result.joint_angles[:, 0], state.joint_angles)


def test_mpc_plan_segment_keeps_frame_zero_foot_state_anchor() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=2, horizon=25)
    cfg.runtime.optimize_steps = 1
    current_foot = state.foot_pos.clone()
    current_foot[:, :, :2] += torch.tensor(
        [[[0.03, 0.01], [-0.04, 0.02], [0.02, -0.03], [-0.01, -0.02]]],
        dtype=torch.float32,
    )
    state = MpcRobotState(
        root_pos=state.root_pos,
        root_rpy=state.root_rpy,
        foot_pos=current_foot,
        joint_angles=state.joint_angles,
    )

    result = plan_segment(terrain, state, command, cfg=cfg)

    torch.testing.assert_close(result.foot_pos[:, 0], state.foot_pos, atol=1.0e-6, rtol=1.0e-6)










def test_mpc_runtime_defaults_to_deterministic_replan_phase_with_task_override() -> None:
    cfg = MpcPlannerCfg()
    assert cfg.runtime.randomize_replan_phase is False

    overridden = planner_cfg_from_task_cfg(SimpleNamespace(mpc_randomize_replan_phase=True))
    assert overridden.runtime.randomize_replan_phase is True


def test_task_mpc_planner_cfg_is_single_source_when_present() -> None:
    task_mpc = MpcPlannerCfg()
    task_mpc.runtime.horizon_steps = 25
    task_mpc.runtime.replan_interval_steps = 25
    task_mpc.runtime.dt = 0.02
    task_mpc.runtime.parallel_plan_batch_size = 64
    task_mpc.diagnostics.emit_runtime_counters = False

    cfg = planner_cfg_from_task_cfg(
        SimpleNamespace(
            mpc_planner_cfg=task_mpc,
            reference_trajectory_horizon=9,
            reference_replan_interval_steps=7,
            plan_dt=0.05,
            mpc_parallel_plan_batch_size=3,
            mpc_diagnostics_emit_runtime_counters=True,
        )
    )

    assert cfg.runtime.horizon_steps == 25
    assert cfg.runtime.replan_interval_steps == 25
    assert cfg.runtime.dt == pytest.approx(0.02)
    assert cfg.runtime.parallel_plan_batch_size == 64
    assert cfg.diagnostics.emit_runtime_counters is False


def test_mpc_semantic_policy_routes_low_small_by_command_type_and_high_large_to_avoidance() -> None:
    low_terrain, state, forward_cmd, cfg = _semantic_obstacle_inputs(
        obstacle_id=1,
        obstacle_height=0.12,
        command=torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32),
    )
    mixed_cmd = torch.tensor([[0.50, 0.25, 1.0]], dtype=torch.float32)
    high_terrain, _, _, _ = _semantic_obstacle_inputs(obstacle_id=1, obstacle_height=0.46, command=forward_cmd)
    large_terrain, _, _, _ = _semantic_obstacle_inputs(obstacle_id=2, obstacle_height=0.28, command=forward_cmd)

    low_forward = classify_semantic_obstacle_mode(low_terrain, state, forward_cmd, cfg)
    low_mixed = classify_semantic_obstacle_mode(low_terrain, state, mixed_cmd, cfg)
    high_small = classify_semantic_obstacle_mode(high_terrain, state, forward_cmd, cfg)
    large = classify_semantic_obstacle_mode(large_terrain, state, forward_cmd, cfg)

    assert low_forward.mode.tolist() == [int(SemanticObstacleMode.LOW_SMALL_FORWARD)]
    assert low_mixed.mode.tolist() == [int(SemanticObstacleMode.LOW_SMALL_MIXED)]
    assert high_small.mode.tolist() == [int(SemanticObstacleMode.HIGH_OR_LARGE_AVOID)]
    assert large.mode.tolist() == [int(SemanticObstacleMode.HIGH_OR_LARGE_AVOID)]


def test_mpc_nominal_command_shaping_reduces_forward_and_adds_lateral_only_for_high_large() -> None:
    low_terrain, state, command, cfg = _semantic_obstacle_inputs(
        obstacle_id=1,
        obstacle_height=0.12,
        command=torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32),
    )
    large_terrain, _, _, _ = _semantic_obstacle_inputs(obstacle_id=2, obstacle_height=0.28, command=command)

    low_shaped, low_diag = shape_nominal_command_for_semantic_obstacles(low_terrain, state, command, cfg)
    large_shaped, large_diag = shape_nominal_command_for_semantic_obstacles(large_terrain, state, command, cfg)

    torch.testing.assert_close(low_shaped, command)
    assert low_diag.command_shaped.tolist() == [False]
    assert float(large_shaped[0, 0]) < float(command[0, 0])
    assert abs(float(large_shaped[0, 1])) >= 0.20
    torch.testing.assert_close(large_shaped[:, 2], command[:, 2], atol=0.0, rtol=0.0)
    assert large_diag.command_shaped.tolist() == [True]


def test_mpc_low_small_foot_crossing_loss_penalizes_stance_and_touchdown_on_low_small() -> None:
    terrain, state, command, cfg = _semantic_obstacle_inputs(obstacle_id=1, obstacle_height=0.12)
    horizon = 8
    root = state.root_pos[:, None, :].expand(1, horizon, 3).clone()
    rpy = state.root_rpy[:, None, :].expand(1, horizon, 3).clone()
    foot_off = state.foot_pos[:, None, :, :].expand(1, horizon, 4, 3).clone()
    foot_on = foot_off.clone()
    foot_on[:, :, 0, :] = torch.tensor([0.40, 0.0, 0.0], dtype=torch.float32)
    contact = torch.ones((1, horizon, 4), dtype=torch.float32)
    swing = torch.zeros_like(contact)
    decoded_off = DecodedTrajectoryStub(root, rpy, foot_off, torch.zeros((1, 4)), torch.ones((1, 4)) * 0.4, torch.zeros((1, 4)), torch.ones((1, 4)), swing, contact)
    decoded_on = DecodedTrajectoryStub(root, rpy, foot_on, torch.zeros((1, 4)), torch.ones((1, 4)) * 0.4, torch.zeros((1, 4)), torch.ones((1, 4)), swing, contact)

    off_loss = low_small_foot_crossing_loss(
        terrain,
        decoded_off,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
        high_small_relative_height_m=cfg.losses.low_small_crossing.high_small_relative_height_m,
        contact_threshold=cfg.runtime.contact_threshold,
    )
    on_loss = low_small_foot_crossing_loss(
        terrain,
        decoded_on,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
        high_small_relative_height_m=cfg.losses.low_small_crossing.high_small_relative_height_m,
        contact_threshold=cfg.runtime.contact_threshold,
    )

    assert float(on_loss[0]) > float(off_loss[0]) + 0.5


def test_mpc_low_small_foot_over_loss_penalizes_side_detour_more_than_swing_over() -> None:
    terrain, state, command, cfg = _semantic_obstacle_inputs(obstacle_id=1, obstacle_height=0.16)
    horizon = 8
    root = state.root_pos[:, None, :].expand(1, horizon, 3).clone()
    root[:, :, 0] = torch.linspace(0.05, 0.75, horizon)
    rpy = state.root_rpy[:, None, :].expand(1, horizon, 3).clone()
    foot_over = state.foot_pos[:, None, :, :].expand(1, horizon, 4, 3).clone()
    foot_over[:, :, 0, :] = torch.tensor([0.40, 0.0, 0.24], dtype=torch.float32)
    foot_side = foot_over.clone()
    foot_side[:, :, 0, 1] = 0.18
    contact = torch.ones((1, horizon, 4), dtype=torch.float32)
    contact[:, :, 0] = 0.0
    swing = 1.0 - contact
    decoded_over = DecodedTrajectoryStub(
        root,
        rpy,
        foot_over,
        torch.zeros((1, 4)),
        torch.ones((1, 4)) * 0.4,
        torch.zeros((1, 4)),
        torch.ones((1, 4)),
        swing,
        contact,
    )
    decoded_side = DecodedTrajectoryStub(
        root,
        rpy,
        foot_side,
        torch.zeros((1, 4)),
        torch.ones((1, 4)) * 0.4,
        torch.zeros((1, 4)),
        torch.ones((1, 4)),
        swing,
        contact,
    )

    over_loss = low_small_foot_over_loss(
        terrain,
        decoded_over,
        command,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
    )
    side_loss = low_small_foot_over_loss(
        terrain,
        decoded_side,
        command,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
    )

    assert float(side_loss[0]) > float(over_loss[0]) + 0.5


def test_mpc_low_small_foot_over_path_curve_penalizes_jump_into_obstacle_window() -> None:
    terrain, state, command, cfg = _semantic_obstacle_inputs(obstacle_id=1, obstacle_height=0.16)
    horizon = 4
    root = torch.tensor(
        [[[0.05, 0.0, 0.34], [0.30, 0.0, 0.34], [0.50, 0.0, 0.34], [0.75, 0.0, 0.34]]],
        dtype=torch.float32,
    )
    rpy = state.root_rpy[:, None, :].expand(1, horizon, 3).clone()
    smooth_foot = state.foot_pos[:, None, :, :].expand(1, horizon, 4, 3).clone()
    smooth_foot[:, :, 0, :] = torch.tensor(
        [[[0.10, 0.0, 0.21], [0.30, 0.0, 0.27], [0.50, 0.0, 0.27], [0.70, 0.0, 0.21]]],
        dtype=torch.float32,
    )
    jump_foot = smooth_foot.clone()
    jump_foot[:, :, 0, :] = torch.tensor(
        [[[0.00, 0.0, 0.21], [0.40, 0.0, 0.27], [0.40, 0.0, 0.27], [0.80, 0.0, 0.21]]],
        dtype=torch.float32,
    )
    contact = torch.ones((1, horizon, 4), dtype=torch.float32)
    contact[:, :, 0] = 0.0
    swing = 1.0 - contact
    common = dict(
        root_pos=root,
        root_rpy=rpy,
        swing_center=torch.zeros((1, 4)),
        swing_width=torch.ones((1, 4)) * 0.4,
        swing_start=torch.zeros((1, 4)),
        swing_end=torch.ones((1, 4)),
        swing_prob=swing,
        contact_prob=contact,
    )
    decoded_smooth = DecodedTrajectoryStub(foot_pos=smooth_foot, **common)
    decoded_jump = DecodedTrajectoryStub(foot_pos=jump_foot, **common)

    smooth_loss = low_small_foot_over_loss(
        terrain,
        decoded_smooth,
        command,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
        xy_weight=0.0,
        direct_xy_weight=0.0,
        z_weight=0.0,
        ineligible_penalty=0.0,
        time_gate_penalty=0.0,
        path_curve_weight=100.0,
        path_curve_z_weight=100.0,
        path_curve_window_m=0.30,
        path_curve_body_yaw=False,
    )
    jump_loss = low_small_foot_over_loss(
        terrain,
        decoded_jump,
        command,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
        xy_weight=0.0,
        direct_xy_weight=0.0,
        z_weight=0.0,
        ineligible_penalty=0.0,
        time_gate_penalty=0.0,
        path_curve_weight=100.0,
        path_curve_z_weight=100.0,
        path_curve_window_m=0.30,
        path_curve_body_yaw=False,
    )

    assert float(jump_loss[0]) > float(smooth_loss[0]) + 0.5


def test_mpc_low_small_foot_over_loss_is_zero_for_high_small_and_large() -> None:
    high_terrain, state, command, cfg = _semantic_obstacle_inputs(obstacle_id=1, obstacle_height=0.46)
    large_terrain, _, _, _ = _semantic_obstacle_inputs(obstacle_id=2, obstacle_height=0.28)
    horizon = 8
    root = state.root_pos[:, None, :].expand(1, horizon, 3).clone()
    rpy = state.root_rpy[:, None, :].expand(1, horizon, 3).clone()
    foot = state.foot_pos[:, None, :, :].expand(1, horizon, 4, 3).clone()
    foot[:, :, 0, :] = torch.tensor([0.40, 0.0, 0.24], dtype=torch.float32)
    contact = torch.ones((1, horizon, 4), dtype=torch.float32)
    contact[:, :, 0] = 0.0
    swing = 1.0 - contact
    decoded = DecodedTrajectoryStub(
        root,
        rpy,
        foot,
        torch.zeros((1, 4)),
        torch.ones((1, 4)) * 0.4,
        torch.zeros((1, 4)),
        torch.ones((1, 4)),
        swing,
        contact,
    )

    high_loss = low_small_foot_over_loss(
        high_terrain,
        decoded,
        command,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
    )
    large_loss = low_small_foot_over_loss(
        large_terrain,
        decoded,
        command,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
    )

    torch.testing.assert_close(high_loss, torch.zeros_like(high_loss))
    torch.testing.assert_close(large_loss, torch.zeros_like(large_loss))


def test_mpc_low_small_stepcap_continuity_loss_is_gated_to_mixed_low_small_and_penalizes_spikes() -> None:
    terrain, state, _, cfg = _semantic_obstacle_inputs(obstacle_id=1, obstacle_height=0.12)
    horizon = 10
    root = state.root_pos[:, None, :].expand(1, horizon, 3).clone()
    rpy = state.root_rpy[:, None, :].expand(1, horizon, 3).clone()
    foot_smooth = state.foot_pos[:, None, :, :].expand(1, horizon, 4, 3).clone()
    foot_spike = foot_smooth.clone()
    foot_spike[:, 5, 0, 0] += 0.45
    contact = torch.full((1, horizon, 4), 0.5, dtype=torch.float32)
    swing = 1.0 - contact
    decoded_smooth = DecodedTrajectoryStub(root, rpy, foot_smooth, torch.zeros((1, 4)), torch.ones((1, 4)) * 0.4, torch.zeros((1, 4)), torch.ones((1, 4)), swing, contact)
    decoded_spike = DecodedTrajectoryStub(root, rpy, foot_spike, torch.zeros((1, 4)), torch.ones((1, 4)) * 0.4, torch.zeros((1, 4)), torch.ones((1, 4)), swing, contact)
    mixed_cmd = torch.tensor([[0.50, 0.25, 1.0]], dtype=torch.float32)
    forward_cmd = torch.tensor([[0.50, 0.0, 0.0]], dtype=torch.float32)

    smooth_loss = low_small_stepcap_continuity_loss(terrain, decoded_smooth, state, mixed_cmd, cfg)
    spike_loss = low_small_stepcap_continuity_loss(terrain, decoded_spike, state, mixed_cmd, cfg)
    forward_loss = low_small_stepcap_continuity_loss(terrain, decoded_spike, state, forward_cmd, cfg)

    assert float(spike_loss[0]) > float(smooth_loss[0]) + 10.0
    torch.testing.assert_close(forward_loss, torch.zeros_like(forward_loss))


def test_mpc_high_large_stepcap_continuity_loss_is_gated_to_high_large_obstacles() -> None:
    large_terrain, state, command, cfg = _semantic_obstacle_inputs(obstacle_id=2, obstacle_height=0.28)
    low_terrain, _, _, _ = _semantic_obstacle_inputs(obstacle_id=1, obstacle_height=0.12)
    horizon = 10
    root = state.root_pos[:, None, :].expand(1, horizon, 3).clone()
    rpy = state.root_rpy[:, None, :].expand(1, horizon, 3).clone()
    foot_smooth = state.foot_pos[:, None, :, :].expand(1, horizon, 4, 3).clone()
    foot_spike = foot_smooth.clone()
    foot_spike[:, 5, 0, 0] += 0.45
    contact = torch.full((1, horizon, 4), 0.5, dtype=torch.float32)
    swing = 1.0 - contact
    decoded_smooth = DecodedTrajectoryStub(root, rpy, foot_smooth, torch.zeros((1, 4)), torch.ones((1, 4)) * 0.4, torch.zeros((1, 4)), torch.ones((1, 4)), swing, contact)
    decoded_spike = DecodedTrajectoryStub(root, rpy, foot_spike, torch.zeros((1, 4)), torch.ones((1, 4)) * 0.4, torch.zeros((1, 4)), torch.ones((1, 4)), swing, contact)

    smooth_loss = high_large_stepcap_continuity_loss(large_terrain, decoded_smooth, command, cfg)
    spike_loss = high_large_stepcap_continuity_loss(large_terrain, decoded_spike, command, cfg)
    low_loss = high_large_stepcap_continuity_loss(low_terrain, decoded_spike, command, cfg)
    mixed_loss = high_large_stepcap_continuity_loss(
        large_terrain,
        decoded_spike,
        torch.tensor([[0.50, 0.25, 1.0]], dtype=torch.float32),
        cfg,
    )

    assert float(spike_loss[0]) > float(smooth_loss[0]) + 5.0
    torch.testing.assert_close(low_loss, torch.zeros_like(low_loss))
    torch.testing.assert_close(mixed_loss, torch.zeros_like(mixed_loss))






def test_mpc_touchdown_semantic_loss_penalizes_small_and_large_obstacles() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 2] = 1
    semantic[:, 2, 3] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1, 1), world_y_range=(-1, 1))
    touchdown_xy = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32)
    touchdown_z = torch.zeros((1, 4), dtype=torch.float32)

    loss = touchdown_semantic_loss(terrain, touchdown_xy, touchdown_z, small_weight=10.0, large_weight=50.0)

    assert loss.shape == (1,)
    assert float(loss[0]) > 0.0


def test_mpc_heightfield_collision_losses_penalize_body_knee_and_shank() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    height[:, 2, 2] = 0.20
    terrain = MpcPlannerTerrain(
        height_map=height,
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    root = torch.zeros((1, 2, 3), dtype=torch.float32)
    rpy = torch.zeros_like(root)
    root[..., 2] = 0.22
    knee = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    knee[..., 2] = 0.21
    shank = knee.unsqueeze(-2).expand(1, 2, 4, 2, 3).clone()

    body_loss = body_heightfield_collision_loss(
        terrain,
        root,
        rpy,
        bottom_offset_z=-0.18,
        margin_m=0.04,
        stencil_xy=((0.0, 0.0),),
    )
    leg_loss = knee_shank_heightfield_collision_loss(
        terrain,
        knee,
        shank,
        knee_margin_m=0.04,
        shank_margin_m=0.04,
    )

    assert body_loss.shape == (1,)
    assert leg_loss.shape == (1,)
    assert float(body_loss[0]) > 0.0
    assert float(leg_loss[0]) > 0.0


def test_mpc_leg_collision_loss_amplifies_sparse_shank_collisions() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    knee = torch.zeros((1, 6, 4, 3), dtype=torch.float32)
    knee[..., 2] = 0.10
    shank = torch.zeros((1, 6, 4, 2, 3), dtype=torch.float32)
    shank[..., 2] = 0.10
    shank[:, 4, 1, 0, 2] = -0.01

    loss = knee_shank_heightfield_collision_loss(
        terrain,
        knee,
        shank,
        knee_margin_m=0.0,
        shank_margin_m=0.0,
        worst_deficit_weight=8.0,
    )

    mean_only = ((0.0 - (-0.01)) ** 2) / float(6 * 4 * 2)
    assert float(loss[0]) > mean_only * 8.0


def test_mpc_stance_semantic_loss_penalizes_obstacle_contact_frames() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 2] = 1
    semantic[:, 2, 3] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1.0, 1.0), world_y_range=(-1.0, 1.0))
    foot = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        ]],
        dtype=torch.float32,
    )
    contact = torch.zeros((1, 2, 4), dtype=torch.float32)
    contact[:, :, 0] = 1.0
    contact[:, :, 1] = 1.0

    loss = stance_semantic_obstacle_loss(
        terrain,
        foot,
        contact,
        ground_ids=(0,),
        small_ids=(1,),
        large_ids=(2,),
        small_weight=10.0,
        large_weight=50.0,
    )

    assert loss.shape == (1,)
    assert float(loss[0]) > 10.0


def test_mpc_stance_and_swing_terrain_losses_use_height_map() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1, 1),
        world_y_range=(-1, 1),
    )
    foot = torch.zeros((1, 3, 4, 3), dtype=torch.float32)
    contact = torch.ones((1, 3, 4), dtype=torch.float32)
    swing = torch.ones_like(contact) - contact

    stance_loss = stance_ground_loss(terrain, foot, contact)
    swing_loss = swing_clearance_terrain_loss(terrain, foot, swing, min_clearance_m=0.05)

    assert stance_loss.shape == (1,)
    assert swing_loss.shape == (1,)
    assert torch.isfinite(stance_loss).all()
    assert torch.isfinite(swing_loss).all()


def test_mpc_default_swing_clearance_is_stronger_for_rough_terrain_collisions() -> None:
    cfg = MpcPlannerCfg()

    assert cfg.runtime.optimize_steps == 24
    assert cfg.runtime.lr == pytest.approx(2.0e-2)
    assert cfg.runtime.contact_threshold == pytest.approx(0.40)
    assert cfg.runtime.nominal_swing_height_m == pytest.approx(0.12)
    assert cfg.losses.swing_center_urgency.weight == pytest.approx(1.5)
    assert cfg.losses.swing_clearance_terrain.min_clearance_m == pytest.approx(0.12)
    assert cfg.losses.swing_clearance_terrain.weight == pytest.approx(12.0)
    assert cfg.losses.swing_clearance_terrain.worst_deficit_weight == pytest.approx(12.0)
    assert cfg.losses.swing_clearance_terrain.boundary_min_swing_prob == pytest.approx(0.40)
    assert cfg.losses.swing_clearance_terrain.boundary_weight == pytest.approx(0.50)
    assert cfg.losses.foot_trajectory_regularization.weight == pytest.approx(1.0)
    assert cfg.losses.foot_trajectory_regularization.boundary_weight == pytest.approx(8.0)
    assert cfg.losses.foot_trajectory_regularization.accel_weight == pytest.approx(8.0)
    assert cfg.losses.leg_collision.weight == pytest.approx(16.0)
    assert cfg.losses.leg_collision.knee_margin_m == pytest.approx(0.06)
    assert cfg.losses.leg_collision.shank_margin_m == pytest.approx(0.06)
    assert cfg.losses.leg_collision.worst_deficit_weight == pytest.approx(16.0)


def test_teacher_mpc_semantic_env_raises_fk_body_leg_collision_weight() -> None:
    import ast

    path = REPO_ROOT / "Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assignments: dict[str, list[float]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in {"TeacherElevationTrajectoryMpcSemanticEnvCfg", "TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"}:
            continue
        values: list[float] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                if ast.unparse(target) == "self.mpc_planner_cfg.losses.fk_body_leg_collision.weight":
                    value = ast.literal_eval(child.value)
                    values.append(float(value))
        assignments[node.name] = values

    assert assignments["TeacherElevationTrajectoryMpcSemanticEnvCfg"] == [120.0]
    assert assignments["TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY"] in ([], [120.0])


def test_mpc_swing_clearance_loss_amplifies_sparse_heightfield_collisions() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    foot = torch.zeros((1, 6, 4, 3), dtype=torch.float32)
    foot[..., 2] = 0.10
    foot[:, 4, 1, 2] = -0.02
    swing = torch.zeros((1, 6, 4), dtype=torch.float32)
    swing[:, :, 1] = 1.0

    loss = swing_clearance_terrain_loss(
        terrain,
        foot,
        swing,
        min_clearance_m=0.02,
        worst_deficit_weight=4.0,
    )

    mean_only = ((0.02 - (-0.02)) ** 2) / 6.0
    assert float(loss[0]) > mean_only * 4.0


def test_mpc_swing_clearance_loss_uses_exported_swing_threshold() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    foot = torch.zeros((1, 2, 1, 3), dtype=torch.float32)
    foot[:, :, :, 2] = -0.02
    swing = torch.tensor([[[0.59], [0.61]]], dtype=torch.float32)

    loss = swing_clearance_terrain_loss(
        terrain,
        foot,
        swing,
        min_clearance_m=0.0,
        min_swing_prob=0.60,
        hard_active_weight=True,
    )

    assert loss.shape == (1,)
    assert float(loss[0]) == pytest.approx(0.0004)


def test_mpc_swing_clearance_loss_penalizes_boundary_swing_halo() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 3, 3), dtype=torch.float32),
        semantic_map=torch.zeros((1, 3, 3), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    foot = torch.zeros((1, 3, 1, 3), dtype=torch.float32)
    foot[:, :, :, 2] = -0.02
    swing = torch.tensor([[[0.44], [0.50], [0.61]]], dtype=torch.float32)

    active_only = swing_clearance_terrain_loss(
        terrain,
        foot,
        swing,
        min_clearance_m=0.0,
        min_swing_prob=0.60,
        hard_active_weight=True,
    )
    loss = swing_clearance_terrain_loss(
        terrain,
        foot,
        swing,
        min_clearance_m=0.0,
        min_swing_prob=0.60,
        hard_active_weight=True,
        boundary_min_swing_prob=0.45,
        boundary_weight=0.25,
    )

    expected_boundary = 0.25 * ((0.50 - 0.45) / (0.60 - 0.45)) * 0.0004
    assert float(active_only[0]) == pytest.approx(0.0004)
    assert float(loss[0]) == pytest.approx(0.0004 + expected_boundary)


def test_mpc_stance_ground_loss_is_not_diluted_by_non_contact_frames() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1, 1),
        world_y_range=(-1, 1),
    )
    foot = torch.zeros((1, 25, 4, 3), dtype=torch.float32)
    contact = torch.zeros((1, 25, 4), dtype=torch.float32)
    foot[:, 10, 2, 2] = 0.10
    contact[:, 10, 2] = 1.0

    loss = stance_ground_loss(terrain, foot, contact)

    assert float(loss[0]) > 0.05


def test_mpc_stance_ground_loss_ignores_frames_below_contact_threshold() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1, 1),
        world_y_range=(-1, 1),
    )
    foot = torch.zeros((1, 3, 4, 3), dtype=torch.float32)
    contact = torch.zeros((1, 3, 4), dtype=torch.float32)
    foot[:, 1, 0, 2] = 0.10
    contact[:, 1, 0] = 0.30

    below_threshold = stance_ground_loss(terrain, foot, contact, min_contact_prob=0.40)
    contact[:, 1, 0] = 0.60
    above_threshold = stance_ground_loss(terrain, foot, contact, min_contact_prob=0.40)

    assert float(below_threshold[0]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(above_threshold[0]) > 0.05


def test_mpc_stance_semantic_loss_ignores_frames_below_contact_threshold() -> None:
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1, 1),
        world_y_range=(-1, 1),
    )
    terrain.semantic_map[:, 2, 2] = 1
    foot = torch.zeros((1, 3, 4, 3), dtype=torch.float32)
    contact = torch.zeros((1, 3, 4), dtype=torch.float32)
    contact[:, 1, 0] = 0.30

    below_threshold = stance_semantic_obstacle_loss(
        terrain,
        foot,
        contact,
        ground_ids=(0,),
        small_ids=(1,),
        large_ids=(2,),
        small_weight=10.0,
        large_weight=50.0,
        min_contact_prob=0.40,
    )
    contact[:, 1, 0] = 0.60
    above_threshold = stance_semantic_obstacle_loss(
        terrain,
        foot,
        contact,
        ground_ids=(0,),
        small_ids=(1,),
        large_ids=(2,),
        small_weight=10.0,
        large_weight=50.0,
        min_contact_prob=0.40,
    )

    assert float(below_threshold[0]) == pytest.approx(0.0, abs=1.0e-6)
    assert float(above_threshold[0]) > 5.0


def test_mpc_semantic_contact_avoidance_loss_pushes_contact_prob_off_obstacles() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 2] = 1
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1, 1), world_y_range=(-1, 1))
    foot = torch.zeros((1, 2, 1, 3), dtype=torch.float32)
    contact = torch.tensor([[[0.8], [0.1]]], dtype=torch.float32)

    loss = semantic_contact_avoidance_loss(
        terrain,
        foot,
        contact,
        ground_ids=(0,),
        small_ids=(1,),
        large_ids=(2,),
        small_weight=10.0,
        large_weight=50.0,
        activation_margin=0.05,
    )

    assert float(loss[0]) == pytest.approx((0.8**2 + 0.1**2) / 2.0)


def test_mpc_semantic_contact_avoidance_loss_penalizes_worst_contact_frame() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 2] = 1
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1, 1), world_y_range=(-1, 1))
    foot = torch.zeros((1, 3, 1, 3), dtype=torch.float32)
    contact = torch.tensor([[[0.8], [0.1], [0.0]]], dtype=torch.float32)

    loss = semantic_contact_avoidance_loss(
        terrain,
        foot,
        contact,
        ground_ids=(0,),
        small_ids=(1,),
        large_ids=(2,),
        small_weight=10.0,
        large_weight=50.0,
        activation_margin=0.05,
        worst_contact_weight=3.0,
    )

    mean_term = (0.8**2 + 0.1**2 + 0.0**2) / 3.0
    worst_term = 3.0 * 0.8**2
    assert float(loss[0]) == pytest.approx(mean_term + worst_term)


def test_mpc_semantic_contact_avoidance_loss_has_xy_gradient_from_soft_field() -> None:
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    semantic[:, 4, 4] = 1
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.4, 0.4), world_y_range=(-0.4, 0.4))
    foot = torch.zeros((1, 2, 1, 3), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        foot[:, 0, 0, 0] = 0.08
        foot[:, 1, 0, 0] = 0.28
    contact = torch.tensor([[[0.8], [0.8]]], dtype=torch.float32)

    loss = semantic_contact_avoidance_loss(
        terrain,
        foot,
        contact,
        ground_ids=(0,),
        small_ids=(1,),
        large_ids=(2,),
        small_weight=10.0,
        large_weight=50.0,
        activation_margin=0.05,
        soft_margin_m=0.20,
        soft_field_weight=1.0,
        soft_worst_field_weight=0.0,
    )

    assert float(loss[0]) > 0.0
    loss.sum().backward()
    assert foot.grad is not None
    assert float(torch.abs(foot.grad[..., :2]).sum().item()) > 0.0


def test_mpc_support_stability_uses_contact_threshold_per_leg() -> None:
    cfg = MpcPlannerCfg()
    assert cfg.losses.contact_regularization.min_support_legs == 2

    diffuse = torch.full((1, 3, 4), 0.30, dtype=torch.float32)
    one_leg = torch.tensor([[[0.80, 0.10, 0.10, 0.10]]], dtype=torch.float32).expand(1, 3, 4)
    stable = torch.tensor([[[0.70, 0.70, 0.05, 0.05]]], dtype=torch.float32).expand(1, 3, 4)

    diffuse_loss = support_stability_loss(
        diffuse,
        min_support_legs=cfg.losses.contact_regularization.min_support_legs,
        contact_threshold=cfg.runtime.contact_threshold,
    )
    one_leg_loss = support_stability_loss(
        one_leg,
        min_support_legs=cfg.losses.contact_regularization.min_support_legs,
        contact_threshold=cfg.runtime.contact_threshold,
    )
    stable_loss = support_stability_loss(
        stable,
        min_support_legs=cfg.losses.contact_regularization.min_support_legs,
        contact_threshold=cfg.runtime.contact_threshold,
    )

    assert float(diffuse_loss[0]) == pytest.approx(2.0 * (cfg.runtime.contact_threshold - 0.30))
    assert float(one_leg_loss[0]) == pytest.approx(cfg.runtime.contact_threshold - 0.10)
    assert float(stable_loss[0]) == pytest.approx(0.0, abs=1.0e-6)


def test_mpc_tracking_loss_uses_body_frame_velocity() -> None:
    root_pos = torch.zeros((1, 2, 3), dtype=torch.float32)
    root_rpy = torch.zeros((1, 2, 3), dtype=torch.float32)
    root_rpy[:, :, 2] = 0.5 * torch.pi
    root_pos[:, 1, 1] = 0.02
    command = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    loss = command_tracking_loss(root_pos, root_rpy, command, dt=0.02)

    assert float(loss[0]) < 1e-4


def test_mpc_tracking_loss_honors_velocity_and_yaw_weights() -> None:
    root_pos = torch.zeros((1, 2, 3), dtype=torch.float32)
    root_rpy = torch.zeros((1, 2, 3), dtype=torch.float32)
    root_pos[:, 1, 0] = 0.02
    root_rpy[:, 1, 2] = 0.02
    command = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    yaw_free = command_tracking_loss(root_pos, root_rpy, command, dt=0.02, vel_weight=1.0, yaw_weight=0.0)
    yaw_penalized = command_tracking_loss(root_pos, root_rpy, command, dt=0.02, vel_weight=1.0, yaw_weight=2.0)

    assert float(yaw_free[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(yaw_penalized[0]) > 1.5


def test_mpc_obstacle_risk_scales_use_all_scanner_obstacle_cells() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    height[:, 2, 4] = 0.45
    semantic[:, 2, 4] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1.0, 1.0), world_y_range=(-1.0, 1.0))
    root = torch.zeros((1, 4, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32)

    scales = obstacle_risk_scales(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        linear_corridor_width_m=0.35,
        linear_forward_distance_m=1.0,
        yaw_swept_radius_m=0.35,
        linear_scale_when_blocked=0.5,
        yaw_scale_when_blocked=0.5,
        linear_speed_eps=1.0e-4,
        yaw_speed_eps=1.0e-4,
    )

    assert scales.linear_scale.shape == (1,)
    assert scales.yaw_scale.shape == (1,)
    assert float(scales.linear_scale[0]) == pytest.approx(0.5)
    assert int(scales.linear_trigger_count[0]) > 0
    assert int(scales.trigger_semantic_class[0]) == 2


def test_mpc_obstacle_risk_scales_classify_sparse_small_semantics_by_nearby_height() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 3] = 1
    height[:, 2, 4] = 0.46
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.5, 0.5), world_y_range=(-0.5, 0.5))
    root = torch.zeros((1, 4, 3), dtype=torch.float32)
    root[:, :, 0] = -0.35
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

    scales = obstacle_risk_scales(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        linear_corridor_width_m=0.35,
        linear_forward_distance_m=1.0,
        yaw_swept_radius_m=0.50,
        linear_scale_when_blocked=0.5,
        yaw_scale_when_blocked=0.5,
        linear_speed_eps=1.0e-4,
        yaw_speed_eps=1.0e-4,
    )

    assert float(scales.linear_scale[0]) == pytest.approx(0.5)
    assert int(scales.linear_trigger_count[0]) > 0
    assert int(scales.trigger_semantic_class[0]) == 1


def test_mpc_obstacle_risk_scales_trigger_when_planned_root_path_nears_high_obstacle() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    height[:, 2, 4] = 0.45
    semantic[:, 2, 4] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.5, 0.5), world_y_range=(-0.5, 0.5))
    root = torch.zeros((1, 4, 3), dtype=torch.float32)
    root[:, :, 0] = torch.linspace(-0.35, 0.25, 4)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    rpy[..., 2] = 0.8
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

    scales = obstacle_risk_scales(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        linear_corridor_width_m=0.25,
        linear_forward_distance_m=1.0,
        yaw_swept_radius_m=0.50,
        linear_scale_when_blocked=0.5,
        yaw_scale_when_blocked=0.5,
        linear_speed_eps=1.0e-4,
        yaw_speed_eps=1.0e-4,
    )

    assert float(scales.linear_scale[0]) == pytest.approx(0.5)
    assert int(scales.linear_trigger_count[0]) > 0


def test_mpc_default_obstacle_risk_width_catches_path_near_high_small() -> None:
    cfg = MpcPlannerCfg()
    height = torch.zeros((1, 11, 11), dtype=torch.float32)
    semantic = torch.zeros((1, 11, 11), dtype=torch.long)
    height[:, 9, 5] = 0.45
    semantic[:, 9, 5] = 1
    terrain = MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
    )
    root = torch.zeros((1, 25, 3), dtype=torch.float32)
    root[:, :, 0] = torch.linspace(-0.2, 0.2, 25)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

    scales = obstacle_risk_scales(
        terrain,
        root,
        rpy,
        command,
        small_ids=cfg.losses.touchdown_semantic.small_ids,
        large_ids=cfg.losses.touchdown_semantic.large_ids,
        high_small_relative_height_m=cfg.losses.obstacle_risk.high_small_relative_height_m,
        linear_corridor_width_m=cfg.losses.obstacle_risk.linear_corridor_width_m,
        linear_forward_distance_m=cfg.losses.obstacle_risk.linear_forward_distance_m,
        yaw_swept_radius_m=cfg.losses.obstacle_risk.yaw_swept_radius_m,
        linear_scale_when_blocked=cfg.losses.obstacle_risk.linear_scale_when_blocked,
        yaw_scale_when_blocked=cfg.losses.obstacle_risk.yaw_scale_when_blocked,
        linear_speed_eps=cfg.losses.obstacle_risk.linear_speed_eps,
        yaw_speed_eps=cfg.losses.obstacle_risk.yaw_speed_eps,
    )

    assert cfg.losses.obstacle_risk.linear_corridor_width_m == pytest.approx(0.40)
    assert float(scales.linear_scale[0]) == pytest.approx(0.5)


def test_mpc_obstacle_risk_scales_handle_yaw_only_swept_region() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    height[:, 2, 3] = 0.45
    semantic[:, 2, 3] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1.0, 1.0), world_y_range=(-1.0, 1.0))
    root = torch.zeros((1, 4, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.0, 0.0, 0.4]], dtype=torch.float32)

    scales = obstacle_risk_scales(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        linear_corridor_width_m=0.35,
        linear_forward_distance_m=1.0,
        yaw_swept_radius_m=0.60,
        linear_scale_when_blocked=0.5,
        yaw_scale_when_blocked=0.5,
        linear_speed_eps=1.0e-4,
        yaw_speed_eps=1.0e-4,
    )

    assert float(scales.linear_scale[0]) == pytest.approx(1.0)
    assert float(scales.yaw_scale[0]) == pytest.approx(0.5)
    assert int(scales.yaw_trigger_count[0]) > 0


def test_mpc_high_obstacle_avoidance_loss_pushes_root_laterally() -> None:
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    height[:, 4, 4] = 0.55
    semantic[:, 4, 4] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.4, 0.4), world_y_range=(-0.4, 0.4))
    root_center = torch.zeros((1, 5, 3), dtype=torch.float32, requires_grad=True)
    root_left = torch.zeros((1, 5, 3), dtype=torch.float32)
    with torch.no_grad():
        root_center[..., 0] = torch.linspace(-0.30, 0.10, 5)
        root_center[..., 2] = 0.65
        root_left.copy_(root_center.detach())
        root_left[..., 1] = 0.40
    rpy = torch.zeros_like(root_center)
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

    center_loss = high_obstacle_avoidance_loss(
        terrain,
        root_center,
        rpy,
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.35,
        forward_distance_m=0.80,
        lateral_clearance_m=0.34,
        longitudinal_influence_m=0.45,
        linear_speed_eps=1.0e-4,
    )
    left_loss = high_obstacle_avoidance_loss(
        terrain,
        root_left,
        rpy.detach(),
        command,
        small_ids=(1,),
        large_ids=(2,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.35,
        forward_distance_m=0.80,
        lateral_clearance_m=0.34,
        longitudinal_influence_m=0.45,
        linear_speed_eps=1.0e-4,
    )

    assert float(center_loss[0]) > float(left_loss[0])
    center_loss.sum().backward()
    assert root_center.grad is not None
    assert float(root_center.grad[..., 1].abs().sum().item()) > 0.0




def test_mpc_low_small_crossing_progress_loss_encourages_root_to_pass_low_small_obstacle() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    height[:, 2, 3] = 0.16
    semantic[:, 2, 3] = 1
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.5, 0.5), world_y_range=(-0.5, 0.5))
    root_before = torch.zeros((1, 25, 3), dtype=torch.float32)
    root_before[..., 2] = 0.30
    root_before[:, 0, 0] = -0.35
    root_before[:, -1, 0] = -0.05
    root_after = root_before.clone()
    root_after[:, -1, 0] = 0.33
    rpy = torch.zeros_like(root_before)
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

    before = low_small_crossing_progress_loss(
        terrain,
        root_before,
        rpy,
        command,
        small_ids=(1,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.25,
        forward_distance_m=1.0,
        pass_margin_m=0.06,
        linear_speed_eps=1.0e-4,
    )
    after = low_small_crossing_progress_loss(
        terrain,
        root_after,
        rpy,
        command,
        small_ids=(1,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.25,
        forward_distance_m=1.0,
        pass_margin_m=0.06,
        linear_speed_eps=1.0e-4,
    )

    assert before.shape == (1,)
    assert float(before[0]) > 0.01
    assert float(after[0]) == pytest.approx(0.0, abs=1.0e-6)


def test_mpc_low_small_crossing_progress_loss_ignores_high_small_obstacle() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    height[:, 2, 3] = 0.46
    semantic[:, 2, 3] = 1
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.5, 0.5), world_y_range=(-0.5, 0.5))
    root = torch.zeros((1, 25, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

    loss = low_small_crossing_progress_loss(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.25,
        forward_distance_m=1.0,
        pass_margin_m=0.06,
        linear_speed_eps=1.0e-4,
    )

    assert float(loss[0]) == pytest.approx(0.0, abs=1.0e-6)


def test_mpc_low_small_crossing_ignores_sparse_semantic_small_when_nearby_height_is_too_high() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 3] = 1
    height[:, 2, 4] = 0.46
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.5, 0.5), world_y_range=(-0.5, 0.5))
    root = torch.zeros((1, 25, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    root[:, 0, 0] = -0.35
    root[:, -1, 0] = -0.05
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32)

    loss = low_small_crossing_progress_loss(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.25,
        forward_distance_m=1.0,
        pass_margin_m=0.06,
        linear_speed_eps=1.0e-4,
    )

    assert float(loss[0]) == pytest.approx(0.0, abs=1.0e-6)


def test_mpc_low_small_crossing_progress_loss_accounts_for_visible_obstacle_depth() -> None:
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    height[:, 4, 5] = 0.16
    semantic[:, 4, 5] = 1
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.4, 0.4), world_y_range=(-0.4, 0.4))
    root_partial = torch.zeros((1, 25, 3), dtype=torch.float32)
    root_partial[..., 2] = 0.30
    root_partial[:, 0, 1] = -0.35
    root_partial[:, -1, 1] = -0.08
    root_crossed = root_partial.clone()
    root_crossed[:, -1, 1] = 0.34
    rpy = torch.zeros_like(root_partial)
    command = torch.tensor([[0.0, 0.25, 0.0]], dtype=torch.float32)

    partial = low_small_crossing_progress_loss(
        terrain,
        root_partial,
        rpy,
        command,
        small_ids=(1,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.25,
        forward_distance_m=1.0,
        pass_margin_m=0.06,
        obstacle_depth_m=0.24,
        linear_speed_eps=1.0e-4,
    )
    crossed = low_small_crossing_progress_loss(
        terrain,
        root_crossed,
        rpy,
        command,
        small_ids=(1,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.25,
        forward_distance_m=1.0,
        pass_margin_m=0.06,
        obstacle_depth_m=0.24,
        linear_speed_eps=1.0e-4,
    )

    assert float(partial[0]) > 0.01
    assert float(crossed[0]) == pytest.approx(0.0, abs=1.0e-6)


def test_mpc_low_small_crossing_progress_loss_uses_scanner_positive_y_cells_for_lateral_commands() -> None:
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    height[:, 8, 4] = 0.16
    semantic[:, 8, 4] = 1
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.4, 0.4), world_y_range=(-0.4, 0.4))
    root = torch.zeros((1, 25, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    root[:, 0, 1] = -0.35
    root[:, -1, 1] = -0.08
    rpy = torch.zeros_like(root)
    command = torch.tensor([[0.0, 0.25, 0.0]], dtype=torch.float32)

    loss = low_small_crossing_progress_loss(
        terrain,
        root,
        rpy,
        command,
        small_ids=(1,),
        high_small_relative_height_m=0.30,
        corridor_width_m=0.25,
        forward_distance_m=1.0,
        pass_margin_m=0.06,
        obstacle_depth_m=0.24,
        linear_speed_eps=1.0e-4,
    )

    assert float(loss[0]) > 0.01


def test_mpc_root_support_geometry_losses_are_finite() -> None:
    root = torch.zeros((1, 5, 3), dtype=torch.float32)
    rpy = torch.zeros((1, 5, 3), dtype=torch.float32)
    foot = torch.tensor([[[[0.2, 0.1, 0.0], [0.2, -0.1, 0.0], [-0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]]]], dtype=torch.float32)
    foot = foot.expand(1, 5, 4, 3).contiguous()
    contact = torch.ones((1, 5, 4), dtype=torch.float32)

    center = root_foot_center_loss(root, foot)
    plane = support_plane_roll_pitch_loss(rpy, foot, contact, swing_weight=0.2)

    assert center.shape == (1,)
    assert plane.shape == (1,)
    assert torch.isfinite(center).all()
    assert torch.isfinite(plane).all()


def test_mpc_support_plane_roll_pitch_uses_root_yaw_frame() -> None:
    yaw = torch.tensor(torch.pi / 2.0, dtype=torch.float32)
    pitch = torch.tensor(0.12, dtype=torch.float32)
    body_xy = torch.tensor(
        [[0.2, 0.1], [0.2, -0.1], [-0.2, 0.1], [-0.2, -0.1]],
        dtype=torch.float32,
    )
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    world_xy = torch.stack(
        (cy * body_xy[:, 0] - sy * body_xy[:, 1], sy * body_xy[:, 0] + cy * body_xy[:, 1]),
        dim=-1,
    )
    foot_z = -torch.tan(pitch) * body_xy[:, 0]
    foot = torch.cat((world_xy, foot_z[:, None]), dim=-1).view(1, 1, 4, 3)
    contact = torch.ones((1, 1, 4), dtype=torch.float32)
    matching_rpy = torch.tensor([[[0.0, pitch.item(), yaw.item()]]], dtype=torch.float32)
    wrong_axis_rpy = torch.tensor([[[pitch.item(), 0.0, yaw.item()]]], dtype=torch.float32)

    matching = support_plane_roll_pitch_loss(matching_rpy, foot, contact, swing_weight=0.0)
    wrong_axis = support_plane_roll_pitch_loss(wrong_axis_rpy, foot, contact, swing_weight=0.0)

    assert float(matching[0]) < 1.0e-3
    assert float(wrong_axis[0]) > 5.0e-2


def test_mpc_ik_fk_residual_matches_clamped_output_joint_contract() -> None:
    root = torch.zeros((1, 1, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    foot = torch.tensor(
        [[[[0.30, 0.05, -0.12], [0.19, -0.05, 0.0], [-0.19, 0.05, 0.0], [-0.19, -0.05, 0.0]]]],
        dtype=torch.float32,
    )
    contact = torch.ones((1, 1, 4), dtype=torch.float32)

    residual = ik_fk_residual_loss(root, rpy, foot, contact, contact_weight=2.0)

    assert float(residual[0]) > 0.02


def test_mpc_ik_fk_residual_contact_term_is_not_diluted_by_non_contact_frames() -> None:
    root = torch.zeros((1, 25, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    foot = torch.tensor(
        [[[[0.19, 0.05, 0.0], [0.19, -0.05, 0.0], [-0.19, 0.05, 0.0], [-0.19, -0.05, 0.0]]]],
        dtype=torch.float32,
    ).expand(1, 25, 4, 3).clone()
    contact = torch.zeros((1, 25, 4), dtype=torch.float32)
    foot[:, -1, 0] = torch.tensor([0.75, 0.05, 0.18], dtype=torch.float32)
    contact[:, -1, 0] = 1.0

    residual = ik_fk_residual_loss(root, rpy, foot, contact, contact_weight=2.0)

    assert float(residual[0]) > 0.10


def test_mpc_swing_center_urgency_order_loss_prefers_urgent_pair_early() -> None:
    _, state, _, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float32)
    swing_center = torch.tensor([[0.25, 0.75, 0.75, 0.25]], dtype=torch.float32)
    swing_width = torch.full((1, 4), 0.5, dtype=torch.float32)
    swapped_center = torch.tensor([[0.75, 0.25, 0.25, 0.75]], dtype=torch.float32)
    foot_body = torch.tensor(
        [[[0.35, 0.12, -0.30], [0.05, -0.12, -0.30], [0.05, 0.12, -0.30], [0.35, -0.12, -0.30]]],
        dtype=torch.float32,
    )
    state = MpcRobotState(
        root_pos=state.root_pos[:1],
        root_rpy=torch.zeros((1, 3), dtype=torch.float32),
        foot_pos=foot_body + state.root_pos[:1, None, :],
        joint_angles=state.joint_angles[:1],
    )

    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    nominal = {"touchdown_target_w": state.foot_pos.clone()}

    good = swing_center_urgency_order_loss(swing_center, swing_width, state, command, cfg.runtime, terrain=terrain, nominal=nominal)
    bad = swing_center_urgency_order_loss(swapped_center, swing_width, state, command, cfg.runtime, terrain=terrain, nominal=nominal)

    assert good.shape == (1,)
    assert bad.shape == (1,)
    assert float(good[0]) < float(bad[0])


def test_mpc_diagonal_pair_loss_handles_wraparound_centers() -> None:
    wrapped = torch.tensor([[0.95, 0.45, 0.45, 0.05]], dtype=torch.float32)
    unwrapped = torch.tensor([[0.25, 0.45, 0.45, 0.05]], dtype=torch.float32)
    width = torch.full((1, 4), 0.5, dtype=torch.float32)

    wrapped_loss = diagonal_pair_loss(wrapped, width)
    unwrapped_loss = diagonal_pair_loss(unwrapped, width)

    assert float(wrapped_loss[0]) < float(unwrapped_loss[0])


def test_mpc_swing_center_urgency_uses_touchdown_semantic_proxy() -> None:
    _, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = command[:1].clone()
    command.zero_()
    swing_center = torch.tensor([[0.75, 0.25, 0.25, 0.75]], dtype=torch.float32)
    swing_width = torch.full((1, 4), 0.5, dtype=torch.float32)
    touchdown = torch.tensor([[[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    nominal = {"touchdown_target_w": touchdown}
    clean = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((1, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    obstacle_semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    obstacle_semantic[:, 2, 2] = 2
    obstacle = MpcPlannerTerrain(
        height_map=torch.zeros((1, 5, 5), dtype=torch.float32),
        semantic_map=obstacle_semantic,
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )

    clean_loss = swing_center_urgency_order_loss(swing_center, swing_width, state, command, cfg.runtime, terrain=clean, nominal=nominal)
    obstacle_loss = swing_center_urgency_order_loss(swing_center, swing_width, state, command, cfg.runtime, terrain=obstacle, nominal=nominal)

    assert not torch.allclose(clean_loss, obstacle_loss)


def test_mpc_semantic_obstacle_loss_allows_cleared_swing_over_obstacle() -> None:
    height = torch.zeros((1, 5, 5), dtype=torch.float32)
    semantic = torch.zeros((1, 5, 5), dtype=torch.long)
    semantic[:, 2, 2] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-1, 1), world_y_range=(-1, 1))
    root = torch.zeros((1, 1, 3), dtype=torch.float32)
    rpy = torch.zeros_like(root)
    foot_low = torch.tensor([[[[0.0, 0.0, 0.01], [0.5, 0.0, 0.2], [-0.5, 0.0, 0.2], [0.0, 0.5, 0.2]]]], dtype=torch.float32)
    foot_high = foot_low.clone()
    foot_high[..., 0, 2] = 0.20
    contact = torch.zeros((1, 1, 4), dtype=torch.float32)
    swing = torch.ones_like(contact)

    low = semantic_obstacle_loss(
        terrain,
        root,
        rpy,
        foot_low,
        contact,
        swing,
        small_weight=1.0,
        large_weight=10.0,
        body_weight=0.0,
        foot_weight=1.0,
        body_stencil_radius_m=0.0,
    )
    high = semantic_obstacle_loss(
        terrain,
        root,
        rpy,
        foot_high,
        contact,
        swing,
        small_weight=1.0,
        large_weight=10.0,
        body_weight=0.0,
        foot_weight=1.0,
        body_stencil_radius_m=0.0,
    )

    assert float(low[0]) > float(high[0])


def test_mpc_semantic_obstacle_loss_has_body_xy_gradient_from_soft_field() -> None:
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    semantic[:, 4, 4] = 2
    terrain = MpcPlannerTerrain(height_map=height, semantic_map=semantic, world_x_range=(-0.4, 0.4), world_y_range=(-0.4, 0.4))
    root = torch.zeros((1, 2, 3), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        root[:, 0, 0] = 0.08
        root[:, 1, 0] = 0.28
        root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    foot = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    foot[..., 2] = 0.20
    contact = torch.zeros((1, 2, 4), dtype=torch.float32)
    swing = torch.ones_like(contact)

    loss = semantic_obstacle_loss(
        terrain,
        root,
        rpy,
        foot,
        contact,
        swing,
        small_weight=1.0,
        large_weight=10.0,
        body_weight=1.0,
        foot_weight=0.0,
        body_stencil_radius_m=0.0,
        soft_margin_m=0.20,
        body_soft_field_weight=1.0,
        body_soft_worst_field_weight=0.0,
    )

    assert float(loss[0]) > 0.0
    loss.sum().backward()
    assert root.grad is not None
    assert float(torch.abs(root.grad[..., :2]).sum().item()) > 0.0


def test_mpc_default_semantic_obstacle_loss_ignores_crossable_small_but_keeps_large() -> None:
    cfg = MpcPlannerCfg()
    height = torch.zeros((1, 9, 9), dtype=torch.float32)
    root = torch.zeros((1, 2, 3), dtype=torch.float32)
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    foot = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    foot[..., 0] = 0.35
    foot[..., 2] = 0.20
    contact = torch.zeros((1, 2, 4), dtype=torch.float32)
    swing = torch.ones_like(contact)

    small_semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    small_semantic[:, 4, 4] = 1
    small_terrain = MpcPlannerTerrain(
        height_map=height.clone(),
        semantic_map=small_semantic,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )
    large_semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    large_semantic[:, 4, 4] = 2
    large_terrain = MpcPlannerTerrain(
        height_map=height.clone(),
        semantic_map=large_semantic,
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )

    small_loss = semantic_obstacle_loss(
        small_terrain,
        root,
        rpy,
        foot,
        contact,
        swing,
        small_weight=cfg.losses.semantic_obstacle.small_weight,
        large_weight=cfg.losses.semantic_obstacle.large_weight,
        body_weight=cfg.losses.semantic_obstacle.body_weight,
        foot_weight=cfg.losses.semantic_obstacle.foot_weight,
        body_stencil_radius_m=0.0,
        soft_margin_m=cfg.losses.semantic_obstacle.soft_margin_m,
        body_soft_field_weight=cfg.losses.semantic_obstacle.body_soft_field_weight,
        body_soft_worst_field_weight=cfg.losses.semantic_obstacle.body_soft_worst_field_weight,
        foot_soft_field_weight=cfg.losses.semantic_obstacle.foot_soft_field_weight,
        foot_soft_worst_field_weight=cfg.losses.semantic_obstacle.foot_soft_worst_field_weight,
        high_small_relative_height_m=cfg.losses.semantic_obstacle.high_small_relative_height_m,
    )
    large_loss = semantic_obstacle_loss(
        large_terrain,
        root,
        rpy,
        foot,
        contact,
        swing,
        small_weight=cfg.losses.semantic_obstacle.small_weight,
        large_weight=cfg.losses.semantic_obstacle.large_weight,
        body_weight=cfg.losses.semantic_obstacle.body_weight,
        foot_weight=cfg.losses.semantic_obstacle.foot_weight,
        body_stencil_radius_m=0.0,
        soft_margin_m=cfg.losses.semantic_obstacle.soft_margin_m,
        body_soft_field_weight=cfg.losses.semantic_obstacle.body_soft_field_weight,
        body_soft_worst_field_weight=cfg.losses.semantic_obstacle.body_soft_worst_field_weight,
        foot_soft_field_weight=cfg.losses.semantic_obstacle.foot_soft_field_weight,
        foot_soft_worst_field_weight=cfg.losses.semantic_obstacle.foot_soft_worst_field_weight,
        high_small_relative_height_m=cfg.losses.semantic_obstacle.high_small_relative_height_m,
    )

    assert float(small_loss[0]) == pytest.approx(0.0, abs=1.0e-7)
    assert float(large_loss[0]) > 0.0


def test_mpc_semantic_obstacle_loss_penalizes_high_small_swing_but_ignores_low_small() -> None:
    cfg = MpcPlannerCfg()
    semantic = torch.zeros((1, 9, 9), dtype=torch.long)
    semantic[:, 4, 4] = 1
    low_height = torch.zeros((1, 9, 9), dtype=torch.float32)
    low_height[:, 4, 4] = 0.16
    high_height = torch.zeros((1, 9, 9), dtype=torch.float32)
    high_height[:, 4, 4] = 0.46
    low = MpcPlannerTerrain(
        height_map=low_height,
        semantic_map=semantic.clone(),
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )
    high = MpcPlannerTerrain(
        height_map=high_height,
        semantic_map=semantic.clone(),
        world_x_range=(-0.4, 0.4),
        world_y_range=(-0.4, 0.4),
    )
    root = torch.zeros((1, 1, 3), dtype=torch.float32)
    root[..., 0] = -0.30
    root[..., 2] = 0.30
    rpy = torch.zeros_like(root)
    foot = torch.zeros((1, 1, 4, 3), dtype=torch.float32)
    foot[..., 2] = 0.20
    contact = torch.zeros((1, 1, 4), dtype=torch.float32)
    swing = torch.ones_like(contact)

    low_loss = semantic_obstacle_loss(
        low,
        root,
        rpy,
        foot,
        contact,
        swing,
        small_weight=cfg.losses.semantic_obstacle.small_weight,
        large_weight=cfg.losses.semantic_obstacle.large_weight,
        body_weight=0.0,
        foot_weight=1.0,
        body_stencil_radius_m=0.0,
        high_small_relative_height_m=cfg.losses.semantic_obstacle.high_small_relative_height_m,
    )
    high_loss = semantic_obstacle_loss(
        high,
        root,
        rpy,
        foot,
        contact,
        swing,
        small_weight=cfg.losses.semantic_obstacle.small_weight,
        large_weight=cfg.losses.semantic_obstacle.large_weight,
        body_weight=0.0,
        foot_weight=1.0,
        body_stencil_radius_m=0.0,
        high_small_relative_height_m=cfg.losses.semantic_obstacle.high_small_relative_height_m,
    )

    assert float(low_loss[0]) == pytest.approx(0.0, abs=1.0e-7)
    assert float(high_loss[0]) > 0.0


def test_mpc_backend_has_no_foothold_memory_symbols() -> None:
    root = GO2PVCNN_ROOT / "extension" / "batch_mpc_planner"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    forbidden = [
        "MpcFootholdMemory",
        "_initialize_foothold_memory",
        "_foothold_memory_for",
        "_update_foothold_memory",
        "_stance_anchor_w",
        "_running_foot_rel_body",
        "_yaw_foot_rel_body",
    ]
    for token in forbidden:
        assert token not in source, token


def test_mpc_plan_segment_outputs_grounded_touchdowns_and_locked_stance() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=2, horizon=25)
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.foot_pos.shape == (2, 25, 4, 3)
    assert result.joint_angles.shape == (2, 25, 12)
    assert result.touchdown_seq.shape[0:2] == (2, 4)
    assert result.planned_touchdown_w.shape == (2, 25, 4, 3)
    touchdown = result.planned_touchdown_w[1, 0]
    terrain_z = height_at(terrain, touchdown[None, :, :2])[1]
    torch.testing.assert_close(touchdown[:, 2], terrain_z, atol=1.0e-6, rtol=1.0e-6)
    fk = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    torch.testing.assert_close(result.foot_pos[1, 1:], fk[1, 1:], atol=1.0e-5, rtol=1.0e-5)


def test_mpc_plan_segment_keeps_zero_command_standstill() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.optimize_steps = 1
    state = MpcRobotState(
        root_pos=state.root_pos,
        root_rpy=state.root_rpy,
        foot_pos=state.foot_pos.clone(),
        joint_angles=state.joint_angles,
    )
    state.foot_pos[:, 0, 2] = 0.12
    state.foot_pos[:, 1, 2] = 0.08

    result = plan_segment(terrain, state, command, cfg=cfg)

    torch.testing.assert_close(result.root_pos, state.root_pos[:, None, :].expand_as(result.root_pos))
    torch.testing.assert_close(result.root_rpy, state.root_rpy[:, None, :].expand_as(result.root_rpy))
    torch.testing.assert_close(result.joint_angles, state.joint_angles[:, None, :].expand_as(result.joint_angles))
    assert result.contact_state.all()
    expected_foot = state.foot_pos.clone()
    expected_foot[:, :, 2] = height_at(terrain, expected_foot[:, :, :2])
    torch.testing.assert_close(result.foot_pos, expected_foot[:, None, :, :].expand_as(result.foot_pos))
    torch.testing.assert_close(result.planned_touchdown_w, expected_foot[:, None, :, :].expand_as(result.planned_touchdown_w))


def test_mpc_loss_registry_no_longer_uses_deleted_terms() -> None:
    assert not (GO2PVCNN_ROOT / "extension" / "batch_mpc_planner" / "losses" / "registry.py").exists()


def test_mpc_t302_losses_do_not_introduce_cpu_hot_path_patterns() -> None:
    root = GO2PVCNN_ROOT / "extension" / "batch_mpc_planner"
    files = [
        root / "kinematics.py",
        root / "losses" / "kinematics.py",
        root / "losses" / "terrain_clearance.py",
        root / "losses" / "tracking.py",
        root / "planner.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    forbidden = [
        ".cpu().numpy(",
        ".numpy()",
        "for env_id in range(",
        "for batch_idx in range(",
    ]
    for token in forbidden:
        assert token not in source, token


def test_mpc_loss_breakdown_exposes_continuous_window_terms() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command[:, 0] = 0.2
    cfg.diagnostics.enabled = True
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert PARAMETRIC_LOSS_KEYS.issubset(result.loss_breakdown)


def test_mpc_cost_breakdown_exposes_t302_collision_and_risk_diagnostics() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command[:, 0] = 0.2
    cfg.diagnostics.enabled = False
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert {"cost_total"}.union(PARAMETRIC_LOSS_KEYS).issubset(result.cost_breakdown)


def test_mpc_viewer_adapter_exposes_cost_breakdown_when_diagnostics_disabled() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command[:, 0] = 0.2
    cfg.diagnostics.enabled = False
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)
    viewer_result = _adapt_mpc_result_for_viewer(result)

    assert result.loss_breakdown is None
    assert viewer_result.loss_breakdown is not None
    assert {"cost_total"}.union(PARAMETRIC_LOSS_KEYS).issubset(viewer_result.loss_breakdown)




def test_mpc_foot_trajectory_regularization_penalizes_boundary_and_acceleration_spikes() -> None:
    foot = torch.zeros((1, 5, 1, 3), dtype=torch.float32)
    foot[0, 1, 0, 0] = 1.0
    foot[0, 2, 0, 0] = 2.0
    foot[0, 3, 0, 0] = 10.0
    foot[0, 4, 0, 0] = 11.0
    swing_prob = torch.tensor([[[0.0], [0.0], [1.0], [0.0], [0.0]]], dtype=torch.float32)

    boundary = foot_boundary_smoothness_loss(foot, swing_prob)
    accel = foot_acceleration_smoothness_loss(foot, swing_prob)

    assert float(boundary[0]) == pytest.approx(4.5)
    assert float(accel[0]) == pytest.approx(7.0)




def test_mpc_task_cfg_overrides_foot_trajectory_regularization() -> None:
    cfg = planner_cfg_from_task_cfg(
        _task_cfg(
            mpc_loss_foot_trajectory_regularization_weight=2.0,
            mpc_loss_foot_trajectory_regularization_boundary_weight=3.0,
            mpc_loss_foot_trajectory_regularization_accel_weight=4.0,
        )
    )

    assert cfg.losses.foot_trajectory_regularization.weight == pytest.approx(2.0)
    assert cfg.losses.foot_trajectory_regularization.boundary_weight == pytest.approx(3.0)
    assert cfg.losses.foot_trajectory_regularization.accel_weight == pytest.approx(4.0)


def test_mpc_manager_runtime_counters_emit_when_enabled() -> None:
    cfg = _task_cfg(
        mpc_diagnostics_emit_runtime_counters=True,
        mpc_optimize_steps=0,
    )
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    manager.refresh_from_env(env)
    counters = manager.runtime_counters()

    assert counters["num_envs"] == 3
    assert counters["global_due"] is True
    assert counters["global_due_count"] == 3
    assert 0 <= counters["sampled_plan_count"] <= cfg.mpc_planner_cfg.runtime.parallel_plan_batch_size
    assert counters["max_stale_observed"] >= 0
    assert counters["planner_ms"] >= 0.0
    assert counters["cache_ms"] >= 0.0


def test_mpc_manager_global_sync_runtime_counters_report_sampled_count() -> None:
    cfg = _task_cfg(
        mpc_parallel_plan_batch_size=2,
        mpc_diagnostics_emit_runtime_counters=True,
        mpc_optimize_steps=0,
    )
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    manager.refresh_from_env(env)
    counters = manager.runtime_counters()

    assert counters["num_envs"] == 3
    assert counters["global_due"] is True
    assert counters["global_due_count"] == 3
    assert counters["sampled_plan_count"] == 2


def test_touchdown_event_cap_is_configurable() -> None:
    terrain, state, _command, cfg = _mpc_plan_inputs(batch=1, horizon=6)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    cfg.runtime.touchdown_event_cap = 3

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.touchdown_seq.shape == (1, 4, 3, 3)


def test_task_cfg_can_carry_complete_mpc_planner_cfg() -> None:
    task_mpc = MpcPlannerCfg()
    task_mpc.losses.tracking.weight = 2.5
    task_mpc.losses.tracking.vel_weight = 3.0
    task_mpc.losses.contact_regularization.enabled = False
    task_mpc.runtime.swing_window_min_width = 0.35
    task_mpc.runtime.swing_window_max_width = 0.65
    task_mpc.runtime.swing_window_center_scale = 0.62
    task_mpc.losses.swing_window.weight = 1.7
    task_mpc.losses.diagonal_pair.weight = 1.8
    task_mpc.losses.swing_center_urgency.weight = 2.1
    task_mpc.losses.stance_ground.weight = 3.0
    task_mpc.losses.swing_clearance_terrain.min_clearance_m = 0.06
    task_mpc.losses.touchdown_semantic.large_weight = 80.0
    task_mpc.losses.touchdown_semantic.small_ids = (3,)
    task_mpc.losses.stance_semantic.small_weight = 12.0
    task_mpc.losses.stance_semantic.large_ids = (4, 5)
    task_mpc.losses.semantic_contact_avoid.weight = 7.0
    task_mpc.losses.semantic_contact_avoid.activation_margin = 0.08
    task_mpc.losses.semantic_contact_avoid.worst_contact_weight = 4.0
    task_mpc.losses.semantic_contact_avoid.soft_margin_m = 0.16
    task_mpc.losses.semantic_contact_avoid.soft_field_weight = 3.0
    task_mpc.losses.semantic_contact_avoid.soft_worst_field_weight = 5.0
    task_mpc.losses.semantic_obstacle.soft_margin_m = 0.22
    task_mpc.losses.semantic_obstacle.body_soft_field_weight = 6.0
    task_mpc.losses.semantic_obstacle.body_soft_worst_field_weight = 7.0
    task_mpc.losses.semantic_obstacle.foot_soft_field_weight = 8.0
    task_mpc.losses.semantic_obstacle.foot_soft_worst_field_weight = 9.0
    task_mpc.losses.body_collision.margin_m = 0.07
    task_mpc.losses.leg_collision.shank_sample_count = 3
    task_mpc.losses.obstacle_risk.high_small_relative_height_m = 0.25
    task_mpc.losses.obstacle_risk.linear_scale_when_blocked = 0.4
    task_mpc.losses.obstacle_risk.yaw_scale_when_blocked = 0.6
    task_mpc.losses.low_small_crossing.weight = 9.0
    task_mpc.losses.low_small_crossing.corridor_width_m = 0.33
    task_mpc.losses.low_small_crossing.obstacle_depth_m = 0.31
    task_mpc.losses.high_obstacle_avoidance.weight = 11.0
    task_mpc.losses.high_obstacle_avoidance.lateral_clearance_m = 0.37
    task_mpc.losses.touchdown_surface.max_slope = 0.45
    task_mpc.losses.root_foot_center.weight = 1.3
    task_mpc.losses.root_height.enabled = False
    task_mpc.losses.root_height.weight = 4.2
    task_mpc.losses.support_plane_rp.swing_weight = 0.15
    task_mpc.runtime.nominal_stride_scale = 0.6
    task_mpc.runtime.nominal_swing_height_m = 0.12
    task_mpc.runtime.nominal_yaw_stride_scale = 0.55
    task_mpc.losses.kinematics.joint_limit_margin_rad = 0.14
    task_mpc.runtime.parallel_plan_batch_size = 123
    task_mpc.diagnostics.enabled = True
    task_mpc.diagnostics.emit_viewer_fields = False
    task_cfg = SimpleNamespace(mpc_planner_cfg=task_mpc)
    cfg = planner_cfg_from_task_cfg(task_cfg)

    assert cfg.losses.tracking.weight == pytest.approx(2.5)
    assert cfg.losses.tracking.vel_weight == pytest.approx(3.0)
    assert cfg.losses.contact_regularization.enabled is False
    assert cfg.runtime.swing_window_min_width == pytest.approx(0.35)
    assert cfg.runtime.swing_window_max_width == pytest.approx(0.65)
    assert cfg.runtime.swing_window_center_scale == pytest.approx(0.62)
    assert cfg.losses.swing_window.weight == pytest.approx(1.7)
    assert cfg.losses.diagonal_pair.weight == pytest.approx(1.8)
    assert cfg.losses.swing_center_urgency.weight == pytest.approx(2.1)
    assert cfg.losses.stance_ground.weight == pytest.approx(3.0)
    assert cfg.losses.swing_clearance_terrain.min_clearance_m == pytest.approx(0.06)
    assert cfg.losses.touchdown_semantic.large_weight == pytest.approx(80.0)
    assert cfg.losses.touchdown_semantic.small_ids == (3,)
    assert cfg.losses.stance_semantic.small_weight == pytest.approx(12.0)
    assert cfg.losses.stance_semantic.large_ids == (4, 5)
    assert cfg.losses.semantic_contact_avoid.weight == pytest.approx(7.0)
    assert cfg.losses.semantic_contact_avoid.activation_margin == pytest.approx(0.08)
    assert cfg.losses.semantic_contact_avoid.worst_contact_weight == pytest.approx(4.0)
    assert cfg.losses.semantic_contact_avoid.soft_margin_m == pytest.approx(0.16)
    assert cfg.losses.semantic_contact_avoid.soft_field_weight == pytest.approx(3.0)
    assert cfg.losses.semantic_contact_avoid.soft_worst_field_weight == pytest.approx(5.0)
    assert cfg.losses.semantic_obstacle.soft_margin_m == pytest.approx(0.22)
    assert cfg.losses.semantic_obstacle.body_soft_field_weight == pytest.approx(6.0)
    assert cfg.losses.semantic_obstacle.body_soft_worst_field_weight == pytest.approx(7.0)
    assert cfg.losses.semantic_obstacle.foot_soft_field_weight == pytest.approx(8.0)
    assert cfg.losses.semantic_obstacle.foot_soft_worst_field_weight == pytest.approx(9.0)
    assert cfg.losses.body_collision.margin_m == pytest.approx(0.07)
    assert cfg.losses.leg_collision.shank_sample_count == 3
    assert cfg.losses.obstacle_risk.high_small_relative_height_m == pytest.approx(0.25)
    assert cfg.losses.obstacle_risk.linear_scale_when_blocked == pytest.approx(0.4)
    assert cfg.losses.obstacle_risk.yaw_scale_when_blocked == pytest.approx(0.6)
    assert cfg.losses.low_small_crossing.weight == pytest.approx(9.0)
    assert cfg.losses.low_small_crossing.corridor_width_m == pytest.approx(0.33)
    assert cfg.losses.low_small_crossing.obstacle_depth_m == pytest.approx(0.31)
    assert cfg.losses.high_obstacle_avoidance.weight == pytest.approx(11.0)
    assert cfg.losses.high_obstacle_avoidance.lateral_clearance_m == pytest.approx(0.37)
    assert cfg.losses.touchdown_surface.max_slope == pytest.approx(0.45)
    assert cfg.losses.root_foot_center.weight == pytest.approx(1.3)
    assert cfg.losses.root_height.enabled is False
    assert cfg.losses.root_height.weight == pytest.approx(4.2)
    assert cfg.losses.support_plane_rp.swing_weight == pytest.approx(0.15)
    assert cfg.runtime.nominal_stride_scale == pytest.approx(0.6)
    assert cfg.runtime.nominal_swing_height_m == pytest.approx(0.12)
    assert cfg.runtime.nominal_yaw_stride_scale == pytest.approx(0.55)
    assert cfg.losses.kinematics.joint_limit_margin_rad == pytest.approx(0.14)
    assert cfg.runtime.parallel_plan_batch_size == 123
    assert cfg.diagnostics.enabled is True
    assert cfg.diagnostics.emit_viewer_fields is False


def test_mpc_default_ik_fk_residual_weight_matches_runtime_acceptance() -> None:
    cfg = MpcPlannerCfg()

    assert cfg.losses.ik_fk_residual.weight == pytest.approx(8.0)


def test_mpc_plan_segment_cuda_path_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    device = torch.device("cuda")
    terrain, state, command, cfg = _mpc_plan_inputs(batch=2, horizon=6)
    state_cuda = MpcRobotState(
        root_pos=state.root_pos.to(device),
        root_rpy=state.root_rpy.to(device),
        foot_pos=state.foot_pos.to(device),
        joint_angles=state.joint_angles.to(device),
    )
    terrain_cuda = MpcPlannerTerrain(
        height_map=terrain.height_map.to(device),
        semantic_map=terrain.semantic_map.to(device) if terrain.semantic_map is not None else None,
        world_x_range=terrain.world_x_range,
        world_y_range=terrain.world_y_range,
    )
    result = plan_segment(terrain_cuda, state_cuda, command.to(device), cfg=cfg)

    assert result.root_pos.device.type == "cuda"
    assert result.contact_state.device.type == "cuda"


def test_mpc_plan_segment_runs_under_inference_mode_when_optimize_steps_positive() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=2, horizon=6)
    cfg.runtime.optimize_steps = 1

    with torch.inference_mode():
        result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.root_pos.shape == (2, 6, 3)
    assert torch.isfinite(result.cost_total).all()
    assert torch.isfinite(result.root_pos).all()
    assert torch.isfinite(result.foot_pos).all()
    assert torch.isfinite(result.joint_angles).all()
    assert result.contact_state.any()
    assert torch.logical_not(result.contact_state).any()




def test_mpc_plan_segment_accepts_inputs_created_under_inference_mode() -> None:
    cfg = MpcPlannerCfg()
    cfg.runtime.horizon_steps = 6
    cfg.runtime.optimize_steps = 1
    cfg.diagnostics.enabled = True

    with torch.inference_mode():
        terrain, state, command, _ = _mpc_plan_inputs(batch=2, horizon=6)
        result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.root_pos.shape == (2, 6, 3)
    assert torch.isfinite(result.cost_total).all()
    assert torch.isfinite(result.root_pos).all()


def _install_fake_eval_cfg_import_dependencies(monkeypatch) -> None:
    import copy

    class _Cfg:
        def __init__(self, *args, **kwargs):
            for index, value in enumerate(args):
                setattr(self, f"arg{index}", value)
            if args and "name" not in kwargs:
                setattr(self, "name", args[0])
            for key, value in kwargs.items():
                setattr(self, key, value)

        def replace(self, **kwargs):
            out = copy.deepcopy(self)
            for key, value in kwargs.items():
                setattr(out, key, value)
            return out

    def _configclass(cls):
        def __init__(self, **kwargs):
            for base in reversed(cls.mro()):
                for name, value in base.__dict__.items():
                    if name.startswith("__") or callable(value) or isinstance(
                        value, (staticmethod, classmethod, property, type)
                    ):
                        continue
                    if hasattr(value, "default_factory") and value.default_factory is not MISSING:
                        setattr(self, name, value.default_factory())
                    elif hasattr(value, "default") and value.default is not MISSING:
                        setattr(self, name, copy.deepcopy(value.default))
                    else:
                        setattr(self, name, copy.deepcopy(value))
            for key, value in kwargs.items():
                setattr(self, key, value)
            post_init = getattr(self, "__post_init__", None)
            if post_init is not None:
                post_init()

        cls.__init__ = __init__
        return cls

    def _module(name: str, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    class _ManagerBasedRLEnvCfg:
        def __post_init__(self):
            if not hasattr(self, "sim"):
                self.sim = SimpleNamespace(physx=SimpleNamespace())

    for name in tuple(sys.modules):
        if name.startswith("isaaclab") or name in {
            "go2_pvcnn.tasks",
            "go2_pvcnn.tasks.register_envs",
            "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg",
            "go2_pvcnn.assets",
            "go2_pvcnn.mdp",
            "go2_pvcnn.sensor.semantic_contacter",
            "go2_pvcnn.sensor.semantic_raycaster",
            "extension.mdp.observations",
            "extension.mdp.rewards_reference",
            "extension.mdp.semantic_body_part_clearance",
            "extension.mdp.semantic_contact_rewards",
            "extension.semantic_course",
        }:
            monkeypatch.delitem(sys.modules, name, raising=False)

    sim_module = _module(
        "isaaclab.sim",
        RigidBodyMaterialCfg=_Cfg,
        MdlFileCfg=_Cfg,
        DomeLightCfg=_Cfg,
    )
    terrain_module = _module(
        "isaaclab.terrains",
        TerrainGeneratorCfg=_Cfg,
        MeshPlaneTerrainCfg=_Cfg,
        HfRandomUniformTerrainCfg=_Cfg,
        HfPyramidSlopedTerrainCfg=_Cfg,
        HfInvertedPyramidSlopedTerrainCfg=_Cfg,
        MeshRandomGridTerrainCfg=_Cfg,
        MeshPyramidStairsTerrainCfg=_Cfg,
        MeshInvertedPyramidStairsTerrainCfg=_Cfg,
        TerrainImporterCfg=_Cfg,
    )
    managers_module = _module(
        "isaaclab.managers",
        CurriculumTermCfg=_Cfg,
        EventTermCfg=_Cfg,
        ObservationGroupCfg=object,
        ObservationTermCfg=_Cfg,
        SceneEntityCfg=_Cfg,
        RewardTermCfg=_Cfg,
        TerminationTermCfg=_Cfg,
    )
    envs_module = _module(
        "isaaclab.envs",
        ManagerBasedRLEnvCfg=_ManagerBasedRLEnvCfg,
        mdp=_module(
            "isaaclab.envs.mdp",
            base_ang_vel=lambda *args, **kwargs: None,
            projected_gravity=lambda *args, **kwargs: None,
            joint_pos_rel=lambda *args, **kwargs: None,
            joint_vel_rel=lambda *args, **kwargs: None,
            generated_commands=lambda *args, **kwargs: None,
            last_action=lambda *args, **kwargs: None,
            base_lin_vel=lambda *args, **kwargs: None,
        ),
    )
    sensors_module = _module(
        "isaaclab.sensors",
        ContactSensorCfg=_Cfg,
        patterns=_module("isaaclab.sensors.patterns", GridPatternCfg=_Cfg),
    )
    isaaclab_module = _module(
        "isaaclab",
        sim=sim_module,
        terrains=terrain_module,
        managers=managers_module,
        envs=envs_module,
        sensors=sensors_module,
    )

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)
    monkeypatch.setitem(sys.modules, "isaaclab.terrains", terrain_module)
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.assets",
        _module("isaaclab.assets", ArticulationCfg=_Cfg, AssetBaseCfg=_Cfg),
    )
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs_module)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", envs_module.mdp)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers_module)
    monkeypatch.setitem(sys.modules, "isaaclab.scene", _module("isaaclab.scene", InteractiveSceneCfg=object))
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors.patterns", sensors_module.patterns)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", _module("isaaclab.utils", configclass=_configclass))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.assets", _module("isaaclab.utils.assets", ISAAC_NUCLEUS_DIR="/Nucleus"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.noise", _module("isaaclab.utils.noise", AdditiveUniformNoiseCfg=_Cfg))
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(register=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "go2_pvcnn.assets", _module("go2_pvcnn.assets", UNITREE_GO2_CFG=_Cfg()))
    monkeypatch.setitem(
        sys.modules,
        "go2_pvcnn.mdp",
        _module(
            "go2_pvcnn.mdp",
            UniformLevelVelocityCommandCfg=type("UniformLevelVelocityCommandCfg", (_Cfg,), {"Ranges": _Cfg}),
            JointPositionActionCfg=_Cfg,
            randomize_rigid_body_material=lambda *args, **kwargs: None,
            randomize_rigid_body_mass=lambda *args, **kwargs: None,
            apply_external_force_torque=lambda *args, **kwargs: None,
            reset_root_state_uniform=lambda *args, **kwargs: None,
            reset_joints_by_scale=lambda *args, **kwargs: None,
            push_by_setting_velocity=lambda *args, **kwargs: None,
            track_lin_vel_xy_exp=lambda *args, **kwargs: None,
            track_ang_vel_z_exp=lambda *args, **kwargs: None,
            lin_vel_z_l2=lambda *args, **kwargs: None,
            ang_vel_xy_l2=lambda *args, **kwargs: None,
            joint_vel_l2=lambda *args, **kwargs: None,
            joint_acc_l2=lambda *args, **kwargs: None,
            joint_torques_l2=lambda *args, **kwargs: None,
            action_rate_l2=lambda *args, **kwargs: None,
            joint_pos_limits=lambda *args, **kwargs: None,
            energy=lambda *args, **kwargs: None,
            flat_orientation_l2=lambda *args, **kwargs: None,
            joint_position_penalty=lambda *args, **kwargs: None,
            feet_air_time=lambda *args, **kwargs: None,
            air_time_variance_penalty=lambda *args, **kwargs: None,
            feet_slide=lambda *args, **kwargs: None,
            undesired_contacts=lambda *args, **kwargs: None,
            time_out=lambda *args, **kwargs: None,
            illegal_contact=lambda *args, **kwargs: None,
            bad_orientation=lambda *args, **kwargs: None,
            terrain_levels_vel_semantic_plane_gate=lambda *args, **kwargs: None,
            lin_vel_cmd_levels=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "go2_pvcnn.sensor.semantic_contacter",
        _module("go2_pvcnn.sensor.semantic_contacter", SemanticGlobalContactSensor=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "go2_pvcnn.sensor.semantic_raycaster",
        _module(
            "go2_pvcnn.sensor.semantic_raycaster",
            SemanticGridRayCasterCfg=type("SemanticGridRayCasterCfg", (_Cfg,), {"OffsetCfg": _Cfg}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "extension.mdp.observations",
        _module("extension.mdp.observations", downsampled_elevation_semantic_scan=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "extension.mdp.rewards_reference",
        _module("extension.mdp.rewards_reference", reference_foot_pos_reward=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "extension.mdp.semantic_body_part_clearance",
        _module(
            "extension.mdp.semantic_body_part_clearance",
            semantic_body_part_clearance_reward=lambda *args, **kwargs: None,
            semantic_foot_over_clearance_bonus=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "extension.mdp.semantic_contact_rewards",
        _module(
            "extension.mdp.semantic_contact_rewards",
            semantic_global_contact_collision_reward=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "extension.semantic_course",
        _module(
            "extension.semantic_course",
            SEMANTIC_COURSE_LARGE_ROOT="/World/semantic_course/large",
            SEMANTIC_COURSE_SMALL_ROOT="/World/semantic_course/small",
            SemanticCourseTerrainImporter=object,
        ),
    )


def test_mpc_policy_eval_cfgs_enable_reference_without_changing_play(monkeypatch) -> None:
    _install_fake_eval_cfg_import_dependencies(monkeypatch)

    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY,
        TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg,
        TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg,
    )

    play = TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY()
    assert play.planner_owned_reference_cache is False
    assert play.use_batched_reference_trajectory is False
    assert play.rewards.reference_foot_pos is None
    assert play.scene.semantic_contact_small is None
    assert play.scene.semantic_contact_large is None

    tracking = TeacherElevationTrajectoryMpcSemanticTrackingEvalEnvCfg()
    assert tracking.planner_owned_reference_cache is True
    assert tracking.use_batched_reference_trajectory is True
    assert tracking.planner_backend == "mpc"
    assert tracking.rewards.reference_foot_pos is not None
    assert tracking.scene.semantic_contact_small is not None
    assert tracking.scene.semantic_contact_large is not None
    assert tracking.mpc_planner_cfg.runtime.horizon_steps == 25
    assert tracking.mpc_planner_cfg.runtime.replan_interval_steps == 25
    assert tracking.mpc_planner_cfg.losses.progress.weight > MpcPlannerCfg().losses.progress.weight
    assert tracking.mpc_planner_cfg.losses.swing_direction.weight > MpcPlannerCfg().losses.swing_direction.weight
    assert tracking.semantic_obstacle_curriculum.plane_counts[0].small == 0
    assert tracking.semantic_obstacle_curriculum.plane_counts[0].large == 0
    assert tracking.semantic_obstacle_curriculum.non_plane_counts[0].small == 0
    assert tracking.semantic_obstacle_curriculum.non_plane_counts[0].large == 0
    assert tracking.scene.terrain.semantic_obstacle_curriculum is tracking.semantic_obstacle_curriculum

    collision = TeacherElevationTrajectoryMpcSemanticSmallCollisionEvalEnvCfg()
    assert collision.planner_owned_reference_cache is True
    assert collision.use_batched_reference_trajectory is True
    assert collision.planner_backend == "mpc"
    assert collision.scene.semantic_contact_small is not None
    assert collision.scene.semantic_contact_large is not None
    assert hasattr(collision, "small_collision_eval_small_count_per_tile")
    assert collision.small_collision_eval_small_count_per_tile > 0


def test_flat_small_avoidance_cfg_static_contract(monkeypatch) -> None:
    _install_fake_eval_cfg_import_dependencies(monkeypatch)

    from go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg import (
        TeacherElevationTrajectoryMpcSemanticEnvCfg,
        TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg,
    )

    base = TeacherElevationTrajectoryMpcSemanticEnvCfg()
    cfg = TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg()

    assert isinstance(cfg, TeacherElevationTrajectoryMpcSemanticEnvCfg)
    assert cfg.experiment_name == "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance"
    assert base.curriculum.lin_vel_cmd_levels is not None
    assert cfg.curriculum.lin_vel_cmd_levels is None
    assert cfg.curriculum.terrain_levels is not None
    assert tuple(cfg.commands.base_velocity.ranges.lin_vel_x) == (0.6, 1.0)
    assert tuple(cfg.commands.base_velocity.ranges.lin_vel_y) == (-0.2, 0.2)
    assert tuple(cfg.commands.base_velocity.ranges.ang_vel_z) == (-0.3, 0.3)
    assert getattr(base.rewards, "semantic_body_part_clearance", None) is None
    assert cfg.rewards.semantic_body_part_clearance is not None
    assert cfg.rewards.semantic_body_part_clearance.params["asset_cfg"].name == "robot"
    assert cfg.rewards.semantic_body_part_clearance.params["scanner_cfg"].name == "semantic_height_scanner"
    assert cfg.rewards.semantic_body_part_clearance.params["small_semantic_ids"] == (1,)
    assert cfg.rewards.semantic_body_part_clearance.params["calf_sections"] == 7
    assert cfg.rewards.semantic_body_part_clearance.params["thigh_sections"] == 7
    assert cfg.rewards.semantic_body_part_clearance.params["include_base"] is True
    assert cfg.rewards.semantic_body_part_clearance.params["base_footprint_grid"] == (5, 3)
    assert cfg.rewards.semantic_body_part_clearance.params["clearance_scale"] == 1000.0
    assert cfg.rewards.semantic_foot_over_clearance is not None
    assert cfg.rewards.semantic_foot_over_clearance.weight > 0.0
    assert cfg.rewards.semantic_foot_over_clearance.params["lookahead_m"] == pytest.approx(1.6)
    assert cfg.rewards.semantic_foot_over_clearance.params["corridor_width_m"] == pytest.approx(0.42)
    assert cfg.rewards.semantic_foot_over_clearance.params["clearance_margin_m"] == pytest.approx(0.05)
    assert cfg.rewards.semantic_contact_collision is not None
    assert cfg.rewards.semantic_contact_collision.params["small_weight"] >= 2.0
    assert cfg.rewards.semantic_contact_collision.params["force_scale"] <= 30.0
    assert cfg.rewards.reference_foot_pos is not None
    assert tuple(count.small for count in cfg.semantic_obstacle_curriculum.plane_counts) == (
        8,
        12,
        16,
        24,
        32,
        40,
        52,
        64,
        72,
        80,
    )
    assert tuple(count.large for count in cfg.semantic_obstacle_curriculum.plane_counts) == (0,) * 10
    assert cfg.semantic_obstacle_curriculum.center_safety_half_extent_m == pytest.approx(
        (0.15, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.65, 0.80, 0.85)
    )
    assert cfg.semantic_obstacle_curriculum.min_spacing_clearance_m == pytest.approx(
        (0.08, 0.08, 0.10, 0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15)
    )
    assert tuple(count.small for count in cfg.semantic_obstacle_curriculum.non_plane_counts) == (0,)
    assert tuple(count.large for count in cfg.semantic_obstacle_curriculum.non_plane_counts) == (0,)
    assert tuple(cfg.scene.terrain.terrain_generator.sub_terrains.keys()) == ("flat",)
    assert cfg.scene.terrain.semantic_obstacle_curriculum is cfg.semantic_obstacle_curriculum


def test_flat_small_avoidance_entrypoints_are_registered() -> None:
    train_source = (GO2PVCNN_ROOT / "scripts/train.py").read_text()
    play_source = (GO2PVCNN_ROOT / "scripts/play.py").read_text()
    register_source = (GO2PVCNN_ROOT / "go2_pvcnn/tasks/register_envs.py").read_text()
    agent_source = (GO2PVCNN_ROOT / "agent/train_cfg.py").read_text()
    factory_source = (GO2PVCNN_ROOT / "extension/trajectory_manager_factory.py").read_text()

    experiment_name = "teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance"
    task_id = "Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Flat-Small-Avoidance-Go2-v0"

    assert experiment_name in train_source
    assert experiment_name in play_source
    assert experiment_name in agent_source
    assert experiment_name in factory_source
    assert task_id in register_source
    assert task_id in train_source
    assert task_id.replace("-v0", "-Play-v0") in register_source
    assert task_id.replace("-v0", "-Play-v0") in play_source
    assert "TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg" in register_source
    assert "TeacherElevationTrajectoryMpcSemanticFlatSmallAvoidanceEnvCfg_PLAY" in register_source
