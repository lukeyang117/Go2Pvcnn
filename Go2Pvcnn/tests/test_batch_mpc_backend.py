from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.batch_mpc_planner.config import MpcPlannerCfg, planner_cfg_from_task_cfg
from extension.batch_mpc_planner.losses.contact import support_stability_loss
from extension.batch_mpc_planner.losses.gait_coupling import (
    diagonal_pair_loss,
    root_foot_center_loss,
    support_plane_roll_pitch_loss,
    swing_center_urgency_order_loss,
)
from extension.batch_mpc_planner.losses.terrain_clearance import (
    semantic_obstacle_loss,
    stance_ground_loss,
    swing_clearance_terrain_loss,
    touchdown_semantic_loss,
    touchdown_surface_loss,
)
from extension.batch_mpc_planner.losses.tracking import command_tracking_loss
from extension.batch_mpc_planner.manager import MpcTrajectoryManager
from extension.batch_mpc_planner.planner import plan_segment, sample_touchdown_positions
from extension.batch_mpc_planner.terrain import (
    build_mpc_terrain_from_scanner,
    height_at,
    semantic_at,
    slope_at,
    subset_mpc_terrain,
    support_at,
)
from extension.batch_mpc_planner.losses.kinematics import ik_fk_residual_loss
from extension.batch_mpc_planner.losses.registry import compute_total_loss
from extension.batch_mpc_planner.nominal import build_nominal_trajectory
from extension.batch_mpc_planner.types import MpcPlannerTerrain, MpcRobotState
from extension.batch_mpc_planner.variables import DecodedMpcTrajectory, decode_trajectory, init_optimization_variables
from extension.trajectory_manager_factory import create_trajectory_manager, planner_backend_from_cfg


def _task_cfg(**overrides):
    values = {
        "planner_backend": "mpc",
        "reference_command_name": "base_velocity",
        "reference_height_scanner_name": "height_scanner",
        "reference_trajectory_horizon": 6,
        "reference_replan_interval_steps": 3,
        "plan_dt": 0.02,
        "mpc_max_stale_steps": 6,
        "mpc_max_dirty_envs_per_step": 2,
        "mpc_optimize_steps": 0,
        "mpc_diagnostics_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def _fake_env(*, num_envs: int = 3, device: torch.device | None = None, flatten_ray_hits: bool = False):
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
        ),
        command_manager=_FakeCommandManager(commands),
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long, device=device),
        common_step_counter=0,
        _trajectory_reference_cache=None,
    )


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
    assert manager.horizon_steps() == cfg.reference_trajectory_horizon


def test_factory_rejects_unknown_backend_with_valid_backend_hint() -> None:
    with pytest.raises(ValueError, match="mpc"):
        planner_backend_from_cfg(_task_cfg(planner_backend="dense_mpc"))


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
        "foot_pos_root",
        "contact_state",
        "planned_touchdown_w",
        "phase_index",
        "valid_mask",
    }
    assert current["root_pos_w"].shape == (3, 3)
    assert current["root_quat_w"].shape == (3, 4)
    assert current["joint_angles"].shape == (3, 12)
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


def test_mpc_manager_supports_flattened_scanner_ray_hits_shape() -> None:
    cfg = _task_cfg()
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3, flatten_ray_hits=True)

    cache = manager.refresh_from_env(env)

    assert cache.is_ready()
    assert cache.root_pos_w.shape == (3, 6, 3)


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
    expected_loss_terms = {
        "swing_window",
        "diagonal_pair",
        "swing_center_urgency",
        "stance_ground",
        "swing_clearance_terrain",
        "touchdown_surface",
        "touchdown_semantic",
        "swing_direction",
        "ik_joint_limit",
        "ik_fk_residual",
        "root_foot_center",
        "support_plane_rp",
    }
    assert expected_loss_terms.issubset(result.loss_breakdown)
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


def test_mpc_decode_uses_continuous_swing_window_variables() -> None:
    _, state, command, cfg = _mpc_plan_inputs(batch=2, horizon=25)
    terrain = MpcPlannerTerrain(
        height_map=torch.zeros((2, 5, 5), dtype=torch.float32),
        semantic_map=torch.zeros((2, 5, 5), dtype=torch.long),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )
    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)
    variables = init_optimization_variables(nominal, cfg.runtime)
    decoded = decode_trajectory(nominal, variables, cfg.runtime)

    assert decoded.swing_center.shape == (2, 4)
    assert decoded.swing_width.shape == (2, 4)
    assert decoded.swing_start.shape == (2, 4)
    assert decoded.swing_end.shape == (2, 4)
    assert decoded.swing_prob.shape == (2, 25, 4)
    assert decoded.contact_prob.shape == (2, 25, 4)
    assert torch.all(decoded.swing_width >= cfg.runtime.swing_window_min_width)
    assert torch.all(decoded.swing_width <= cfg.runtime.swing_window_max_width)
    torch.testing.assert_close(
        decoded.swing_prob + decoded.contact_prob,
        torch.ones_like(decoded.swing_prob),
        atol=1e-5,
        rtol=1e-5,
    )


def test_mpc_nominal_integrates_body_frame_command_with_yaw() -> None:
    terrain, state, _, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    state = MpcRobotState(
        root_pos=torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float32),
        root_rpy=torch.tensor([[0.0, 0.0, 0.5 * torch.pi]], dtype=torch.float32),
        foot_pos=state.foot_pos[:1].to(torch.float32),
        joint_angles=state.joint_angles[:1].to(torch.float32),
    )
    command = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32)

    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)

    assert nominal["root_pos"].shape == (1, 25, 3)
    assert nominal["root_pos"][0, -1, 1] > nominal["root_pos"][0, -1, 0]
    torch.testing.assert_close(nominal["root_pos"][0, :, 2], torch.full((25,), 0.3))


def test_mpc_nominal_touchdown_target_uses_swing_time_root_frame() -> None:
    terrain, state, _, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    command = torch.tensor([[0.2, 0.0, 0.5]], dtype=torch.float32)

    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)

    assert "swing_center" in nominal
    assert "swing_width" in nominal
    assert "touchdown_target_w" in nominal
    front_x = nominal["touchdown_target_w"][0, :2, 0]
    rear_x = nominal["touchdown_target_w"][0, 2:, 0]
    assert not torch.allclose(front_x.mean(), rear_x.mean())


def test_mpc_nominal_first_frame_keeps_current_feet_and_diagonal_prior() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.randomize_replan_phase = False

    nominal = build_nominal_trajectory(state, command[:1], terrain, cfg.runtime)

    torch.testing.assert_close(nominal["foot_pos"][0, 0], state.foot_pos[0], atol=1e-6, rtol=1e-6)
    assert nominal["contact_prior"][0, 0].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_mpc_nominal_holds_touchdown_target_after_swing() -> None:
    terrain, state, _, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.randomize_replan_phase = False
    command = torch.tensor([[-0.45, 0.0, 0.0]], dtype=torch.float32)

    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)

    post_touchdown_frame = 14
    for leg_idx in (1, 2):
        torch.testing.assert_close(
            nominal["foot_pos"][0, post_touchdown_frame, leg_idx],
            nominal["touchdown_target_w"][0, leg_idx],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        assert not torch.allclose(
            nominal["foot_pos"][0, post_touchdown_frame, leg_idx],
            state.foot_pos[0, leg_idx],
            atol=1.0e-4,
            rtol=1.0e-4,
        )


def test_mpc_wraparound_touchdown_samples_horizon_end_target() -> None:
    terrain, state, _, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.randomize_replan_phase = False
    command = torch.tensor([[0.45, 0.0, 0.0]], dtype=torch.float32)

    nominal = build_nominal_trajectory(state, command, terrain, cfg.runtime)
    touchdown_w = sample_touchdown_positions(
        nominal["foot_pos"],
        nominal["swing_center"],
        nominal["swing_width"],
    )

    for leg_idx in (0, 3):
        assert nominal["touchdown_phase"][0, leg_idx] >= 1.0 - 1.0e-6
        torch.testing.assert_close(
            nominal["foot_pos"][0, -1, leg_idx],
            nominal["touchdown_target_w"][0, leg_idx],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            touchdown_w[0, leg_idx],
            nominal["touchdown_target_w"][0, leg_idx],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        assert not torch.allclose(
            touchdown_w[0, leg_idx],
            state.foot_pos[0, leg_idx],
            atol=1.0e-4,
            rtol=1.0e-4,
        )


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

    assert float(diffuse_loss[0]) > 0.4
    assert float(one_leg_loss[0]) > 0.3
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


def test_mpc_backend_has_no_foothold_memory_or_output_grounding_symbols() -> None:
    root = GO2PVCNN_ROOT / "extension" / "batch_mpc_planner"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    forbidden = [
        "MpcFootholdMemory",
        "_ground_contact_feet_to_terrain",
        "_initialize_foothold_memory",
        "_foothold_memory_for",
        "_update_foothold_memory",
        "_stance_anchor_w",
        "_running_foot_rel_body",
        "_yaw_foot_rel_body",
    ]
    for token in forbidden:
        assert token not in source, token


def test_mpc_plan_segment_outputs_optimized_feet_without_post_grounding() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.foot_pos.shape == (1, 25, 4, 3)
    assert result.joint_angles.shape == (1, 25, 12)
    assert result.touchdown_seq.shape[0:2] == (1, 4)
    assert result.planned_touchdown_w.shape == (1, 25, 4, 3)


def test_mpc_plan_segment_keeps_zero_command_standstill() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    torch.testing.assert_close(result.root_pos, state.root_pos[:, None, :].expand_as(result.root_pos))
    torch.testing.assert_close(result.root_rpy, state.root_rpy[:, None, :].expand_as(result.root_rpy))
    torch.testing.assert_close(result.foot_pos, state.foot_pos[:, None, :, :].expand_as(result.foot_pos))
    torch.testing.assert_close(result.joint_angles, state.joint_angles[:, None, :].expand_as(result.joint_angles))
    assert result.contact_state.all()
    torch.testing.assert_close(
        result.planned_touchdown_w,
        state.foot_pos[:, None, :, :].expand_as(result.planned_touchdown_w),
    )


def test_mpc_loss_registry_no_longer_uses_deleted_terms() -> None:
    source = (GO2PVCNN_ROOT / "extension" / "batch_mpc_planner" / "losses" / "registry.py").read_text(encoding="utf-8")
    forbidden = [
        "_command_adaptive_weights",
        "contact_schedule_tracking_loss",
        "touchdown_support",
        "obstacle_margin_loss",
        "swing_stride_loss",
    ]
    for token in forbidden:
        assert token not in source, token


def test_mpc_loss_breakdown_exposes_continuous_window_terms() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    cfg.diagnostics.enabled = True
    cfg.runtime.optimize_steps = 1

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    expected = {
        "swing_window",
        "diagonal_pair",
        "swing_center_urgency",
        "stance_ground",
        "swing_clearance_terrain",
        "touchdown_surface",
        "touchdown_semantic",
        "swing_direction",
        "ik_joint_limit",
        "ik_fk_residual",
        "root_foot_center",
        "root_height",
        "support_plane_rp",
    }
    assert expected.issubset(result.loss_breakdown)


def test_mpc_root_height_loss_penalizes_z_drift_from_nominal() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=25)
    nominal = build_nominal_trajectory(state, command[:1], terrain, cfg.runtime)
    variables = init_optimization_variables(nominal, cfg.runtime)
    decoded = decode_trajectory(nominal, variables, cfg.runtime)
    drifted = DecodedMpcTrajectory(
        root_pos=decoded.root_pos + torch.tensor([0.0, 0.0, 0.10], dtype=decoded.root_pos.dtype).view(1, 1, 3),
        root_rpy=decoded.root_rpy,
        foot_pos=decoded.foot_pos,
        swing_center=decoded.swing_center,
        swing_width=decoded.swing_width,
        swing_start=decoded.swing_start,
        swing_end=decoded.swing_end,
        swing_prob=decoded.swing_prob,
        contact_prob=decoded.contact_prob,
    )

    _, _, breakdown = compute_total_loss(drifted, nominal, state, command[:1], terrain, cfg)

    assert "root_height" in breakdown
    assert float(breakdown["root_height"][0]) > 0.05


def test_mpc_manager_runtime_counters_emit_when_enabled() -> None:
    cfg = _task_cfg(
        mpc_diagnostics_emit_runtime_counters=True,
        mpc_max_dirty_envs_per_step=2,
        mpc_optimize_steps=0,
    )
    manager = create_trajectory_manager(cfg, device="cpu")
    env = _fake_env(num_envs=3)

    manager.refresh_from_env(env)
    counters = manager.runtime_counters()

    assert counters["num_envs"] == 3
    assert counters["dirty_count"] >= counters["selected_dirty_count"] >= 0
    assert counters["dirty_backlog"] == counters["dirty_count"] - counters["selected_dirty_count"]
    assert counters["selected_dirty_count"] <= cfg.mpc_max_dirty_envs_per_step
    assert counters["max_stale_observed"] >= 0
    assert counters["planner_ms"] >= 0.0
    assert counters["cache_ms"] >= 0.0


def test_touchdown_event_cap_is_configurable() -> None:
    terrain, state, command, cfg = _mpc_plan_inputs(batch=1, horizon=6)
    cfg.runtime.touchdown_event_cap = 3

    result = plan_segment(terrain, state, command, cfg=cfg)

    assert result.touchdown_seq.shape == (1, 4, 3, 3)


def test_task_cfg_can_override_loss_and_diagnostics_parameters() -> None:
    task_cfg = _task_cfg(
        mpc_loss_tracking_weight=2.5,
        mpc_loss_tracking_vel_weight=3.0,
        mpc_loss_contact_regularization_enabled=False,
        mpc_swing_window_min_width=0.35,
        mpc_swing_window_max_width=0.65,
        mpc_swing_window_center_scale=0.62,
        mpc_loss_swing_window_weight=1.7,
        mpc_loss_diagonal_pair_weight=1.8,
        mpc_loss_swing_center_urgency_weight=2.1,
        mpc_loss_stance_ground_weight=3.0,
        mpc_loss_swing_clearance_terrain_min_clearance_m=0.06,
        mpc_loss_touchdown_semantic_large_weight=80.0,
        mpc_loss_touchdown_surface_max_slope=0.45,
        mpc_loss_root_foot_center_weight=1.3,
        mpc_loss_root_height_enabled=False,
        mpc_loss_root_height_weight=4.2,
        mpc_loss_support_plane_rp_swing_weight=0.15,
        mpc_nominal_stride_scale=0.6,
        mpc_nominal_swing_height_m=0.12,
        mpc_nominal_yaw_stride_scale=0.55,
        mpc_loss_kinematics_joint_limit_margin_rad=0.14,
        mpc_diagnostics_enabled=True,
        mpc_diagnostics_emit_viewer_fields=False,
    )
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
    assert cfg.losses.touchdown_surface.max_slope == pytest.approx(0.45)
    assert cfg.losses.root_foot_center.weight == pytest.approx(1.3)
    assert cfg.losses.root_height.enabled is False
    assert cfg.losses.root_height.weight == pytest.approx(4.2)
    assert cfg.losses.support_plane_rp.swing_weight == pytest.approx(0.15)
    assert cfg.runtime.nominal_stride_scale == pytest.approx(0.6)
    assert cfg.runtime.nominal_swing_height_m == pytest.approx(0.12)
    assert cfg.runtime.nominal_yaw_stride_scale == pytest.approx(0.55)
    assert cfg.losses.kinematics.joint_limit_margin_rad == pytest.approx(0.14)
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
