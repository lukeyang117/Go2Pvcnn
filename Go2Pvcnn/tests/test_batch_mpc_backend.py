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
from extension.batch_mpc_planner.manager import MpcTrajectoryManager
from extension.batch_mpc_planner.planner import plan_segment
from extension.batch_mpc_planner.terrain import build_mpc_terrain_from_scanner, subset_mpc_terrain
from extension.batch_mpc_planner.types import MpcPlannerTerrain, MpcRobotState
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
        data=SimpleNamespace(ray_hits_w=ray_hits_w, semantic_map=semantic_map),
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
    assert "contact_schedule" in result.loss_breakdown
    assert "stance_slip" in result.loss_breakdown
    assert "swing_stride" in result.loss_breakdown
    assert "root_frame_drift" in result.loss_breakdown
    assert "root_frame_follow" in result.loss_breakdown
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
        mpc_loss_contact_schedule_min_support_prob=0.42,
        mpc_loss_obstacle_large_body_margin_m=0.12,
        mpc_loss_progress_min_progress_m=0.09,
        mpc_loss_stance_slip_tolerance_m_per_step=0.006,
        mpc_loss_swing_stride_min_swing_span_m=0.07,
        mpc_loss_swing_stride_command_speed_deadzone_mps=0.11,
        mpc_loss_root_frame_drift_min_rel_m=0.21,
        mpc_loss_root_frame_drift_max_rel_m=0.73,
        mpc_loss_root_frame_follow_rel_change_tolerance_m_per_step=0.031,
        mpc_nominal_stride_scale=0.7,
        mpc_nominal_max_stride_m=0.13,
        mpc_nominal_swing_height_m=0.03,
        mpc_nominal_yaw_stride_scale=1.4,
        mpc_nominal_backward_stride_scale=0.62,
        mpc_nominal_yaw_stride_atten=0.28,
        mpc_loss_kinematics_joint_limit_margin_rad=0.14,
        mpc_diagnostics_enabled=True,
        mpc_diagnostics_emit_viewer_fields=False,
    )
    cfg = planner_cfg_from_task_cfg(task_cfg)

    assert cfg.losses.tracking.weight == pytest.approx(2.5)
    assert cfg.losses.tracking.vel_weight == pytest.approx(3.0)
    assert cfg.losses.contact_regularization.enabled is False
    assert cfg.losses.contact_schedule.min_support_prob == pytest.approx(0.42)
    assert cfg.losses.obstacle_large.body_margin_m == pytest.approx(0.12)
    assert cfg.losses.progress.min_progress_m == pytest.approx(0.09)
    assert cfg.losses.stance_slip.slip_tolerance_m_per_step == pytest.approx(0.006)
    assert cfg.losses.swing_stride.min_swing_span_m == pytest.approx(0.07)
    assert cfg.losses.swing_stride.command_speed_deadzone_mps == pytest.approx(0.11)
    assert cfg.losses.root_frame_drift.min_rel_m == pytest.approx(0.21)
    assert cfg.losses.root_frame_drift.max_rel_m == pytest.approx(0.73)
    assert cfg.losses.root_frame_follow.rel_change_tolerance_m_per_step == pytest.approx(0.031)
    assert cfg.runtime.nominal_stride_scale == pytest.approx(0.7)
    assert cfg.runtime.nominal_max_stride_m == pytest.approx(0.13)
    assert cfg.runtime.nominal_swing_height_m == pytest.approx(0.03)
    assert cfg.runtime.nominal_yaw_stride_scale == pytest.approx(1.4)
    assert cfg.runtime.nominal_backward_stride_scale == pytest.approx(0.62)
    assert cfg.runtime.nominal_yaw_stride_atten == pytest.approx(0.28)
    assert cfg.losses.kinematics.joint_limit_margin_rad == pytest.approx(0.14)
    assert cfg.diagnostics.enabled is True
    assert cfg.diagnostics.emit_viewer_fields is False


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
