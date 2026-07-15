from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.batch_mpc_planner.adapter import mpc_result_to_reference_cache
from extension.batch_mpc_planner.types import MpcPlannerTerrain, MpcRobotState
from extension.trajectory_manager_factory import create_trajectory_manager, planner_backend_from_cfg
from extension.viz.go2_foostep_planner import build_arg_parser
from mpc_rl_epoch_perf_probe import _parse_args as _parse_perf_probe_args


def _flat_terrain(*, batch: int = 2, size: int = 9) -> MpcPlannerTerrain:
    height = torch.zeros((batch, size, size), dtype=torch.float32)
    semantic = torch.zeros((batch, size, size), dtype=torch.long)
    return MpcPlannerTerrain(
        height_map=height,
        semantic_map=semantic,
        world_x_range=(-0.5, 0.5),
        world_y_range=(-0.5, 0.5),
        sensor_pos_w=torch.zeros((batch, 3), dtype=torch.float32),
        sensor_yaw=torch.zeros(batch, dtype=torch.float32),
        is_plane_terrain=torch.ones(batch, dtype=torch.bool),
    )


def _state(*, batch: int = 2) -> MpcRobotState:
    root = torch.zeros((batch, 3), dtype=torch.float32)
    root[:, 2] = 0.32
    foot_offsets = torch.tensor(
        [[0.22, 0.12, 0.0], [0.22, -0.12, 0.0], [-0.20, 0.12, 0.0], [-0.20, -0.12, 0.0]],
        dtype=torch.float32,
    )
    return MpcRobotState(
        root_pos=root,
        root_rpy=torch.zeros((batch, 3), dtype=torch.float32),
        foot_pos=root[:, None, :] + foot_offsets[None, :, :],
        joint_angles=torch.zeros((batch, 12), dtype=torch.float32),
    )


def test_mpc_qp_backend_is_explicitly_accepted_and_has_default_single_iteration() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg

    cfg = SimpleNamespace(planner_backend="mpc_qp", mpc_qp_planner_cfg=MpcQpPlannerCfg())

    assert planner_backend_from_cfg(cfg) == "mpc_qp"
    assert cfg.mpc_qp_planner_cfg.runtime.qp_iterations == 1
    manager = create_trajectory_manager(cfg, device="cpu")
    assert manager.planner_backend == "mpc_qp"


def test_mpc_qp_plan_segment_exports_reference_cache_abi_and_iteration_diagnostics() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=2)
    state = _state(batch=2)
    command = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.20, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    cache = mpc_result_to_reference_cache(result)

    assert result.root_pos.shape == (2, cfg.runtime.horizon_steps, 3)
    assert result.foot_pos.shape == (2, cfg.runtime.horizon_steps, 4, 3)
    assert result.joint_angles.shape == (2, cfg.runtime.horizon_steps, 12)
    assert result.contact_state.shape == (2, cfg.runtime.horizon_steps, 4)
    assert cache.root_pos_w.shape == (2, cfg.runtime.horizon_steps, 3)
    assert torch.isfinite(result.root_pos).all()
    assert torch.isfinite(result.foot_pos).all()
    assert torch.isfinite(result.joint_angles).all()
    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_iterations_configured"], torch.ones(2))
    assert torch.equal(result.loss_breakdown["qp_iterations_executed"], torch.ones(2))
    assert "qp_touchdown_semantic_violation_count" in result.loss_breakdown
    assert "qp_height_violation_max" in result.loss_breakdown
    assert torch.count_nonzero(result.loss_breakdown["qp_touchdown_semantic_violation_count"]) == 0
    assert torch.count_nonzero(result.loss_breakdown["qp_height_violation_max"]) == 0


def test_mpc_qp_repairs_touchdown_that_lands_on_semantic_obstacle() -> None:
    from extension.batch_mpc_planner.terrain import semantic_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    touchdown_xy = baseline.planned_touchdown_w[:, 0, :, :2].detach()
    obstacle_x = touchdown_xy[..., 0].mean()
    obstacle_y = touchdown_xy[..., 1].mean()
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - obstacle_x)).item())
    iy = int(torch.argmin(torch.abs(ys - obstacle_y)).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1

    repaired = plan_segment_qp(terrain, state, command, cfg=cfg)
    touchdown_semantic = semantic_at(terrain, repaired.planned_touchdown_w[:, 0, :, :2])

    assert torch.count_nonzero(touchdown_semantic != 0) == 0
    assert repaired.loss_breakdown is not None
    assert torch.count_nonzero(repaired.loss_breakdown["qp_touchdown_semantic_violation_count"]) == 0


def test_mpc_qp_viewer_argparse_accepts_backend(monkeypatch) -> None:
    fake_isaaclab = ModuleType("isaaclab")
    fake_app = ModuleType("isaaclab.app")

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", type=str, default="cpu")

    fake_app.AppLauncher = _FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    parser = build_arg_parser()
    args = parser.parse_args(["--planner-backend", "mpc_qp"])
    assert args.planner_backend == "mpc_qp"


def test_mpc_qp_viewer_argparse_accepts_and_applies_qp_iterations(monkeypatch) -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.viz import go2_foostep_planner as viewer

    fake_isaaclab = ModuleType("isaaclab")
    fake_app = ModuleType("isaaclab.app")

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", type=str, default="cpu")

    fake_app.AppLauncher = _FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    parser = build_arg_parser()
    args = parser.parse_args(["--planner-backend", "mpc_qp", "--qp-iterations", "3"])
    assert args.planner_backend == "mpc_qp"
    assert args.qp_iterations == 3

    env_cfg = SimpleNamespace(
        planner_backend="mpc_qp",
        mpc_planner_cfg=SimpleNamespace(runtime=SimpleNamespace(horizon_steps=0, replan_interval_steps=0, dt=0.0)),
        mpc_qp_planner_cfg=MpcQpPlannerCfg(),
    )

    viewer._apply_planner_runtime_cli_overrides(env_cfg, args)

    assert env_cfg.mpc_qp_planner_cfg.runtime.qp_iterations == 3


def test_mpc_qp_viewer_argparse_accepts_idle_debug(monkeypatch) -> None:
    fake_isaaclab = ModuleType("isaaclab")
    fake_app = ModuleType("isaaclab.app")

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", type=str, default="cpu")

    fake_app.AppLauncher = _FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    parser = build_arg_parser()
    args = parser.parse_args(["--planner-backend", "mpc_qp", "--idle-debug", "--idle-debug-stride", "3"])

    assert args.idle_debug is True
    assert args.idle_debug_stride == 3


def test_mpc_qp_viewer_argparse_accepts_timing_debug(monkeypatch) -> None:
    fake_isaaclab = ModuleType("isaaclab")
    fake_app = ModuleType("isaaclab.app")

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", type=str, default="cpu")

    fake_app.AppLauncher = _FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    parser = build_arg_parser()
    args = parser.parse_args(["--planner-backend", "mpc_qp", "--timing-debug", "--timing-sync-cuda"])

    assert args.timing_debug is True
    assert args.timing_sync_cuda is True


def test_mpc_qp_viewer_idle_debug_delta_formats_root_and_foot_motion() -> None:
    from extension.viz import go2_foostep_planner as viewer

    previous = viewer._ViewerIdleDebugSample(
        root_pos_w=torch.tensor([[1.0, 2.0, 0.3]], dtype=torch.float64),
        foot_pos_w=torch.zeros((1, 4, 3), dtype=torch.float64),
    )
    current = viewer._ViewerIdleDebugSample(
        root_pos_w=torch.tensor([[1.0, 2.01, 0.3]], dtype=torch.float64),
        foot_pos_w=torch.ones((1, 4, 3), dtype=torch.float64) * 0.02,
    )

    row = viewer._viewer_idle_debug_row(
        cycle=2,
        frame=5,
        command=torch.zeros((1, 3), dtype=torch.float64),
        need_replan=False,
        playback_path="render+scene_sync",
        previous=previous,
        current=current,
    )

    assert row["type"] == "viewer_idle_debug"
    assert row["cycle"] == 2
    assert row["frame"] == 5
    assert row["need_replan"] is False
    assert row["playback_path"] == "render+scene_sync"
    assert abs(float(row["root_delta_m"]) - 0.01) < 1.0e-9
    assert float(row["foot_delta_max_m"]) > 0.0


def test_mpc_qp_viewer_timing_debug_row_includes_viewer_and_qp_timings() -> None:
    from extension.viz import go2_foostep_planner as viewer

    result = SimpleNamespace(
        loss_breakdown={
            "qp_nominal_ms": torch.tensor([1.25], dtype=torch.float32),
            "qp_solve_ms": torch.tensor([2.5], dtype=torch.float32),
            "qp_repair_ms": torch.tensor([0.0], dtype=torch.float32),
            "qp_diagnostics_ms": torch.tensor([0.75], dtype=torch.float32),
            "qp_total_ms": torch.tensor([4.5], dtype=torch.float32),
        }
    )

    row = viewer._viewer_timing_debug_row(
        cycle=3,
        frame=0,
        command=torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32),
        command_changed=True,
        need_replan=True,
        force_zero_hold=False,
        result=result,
        timings_ms={
            "teleop_poll": 0.1,
            "terrain_build": 1.0,
            "plan": 5.0,
            "playback": 6.0,
        },
    )

    assert row["type"] == "viewer_timing_debug"
    assert row["cycle"] == 3
    assert row["command_changed"] is True
    assert row["need_replan"] is True
    assert row["teleop_poll_ms"] == 0.1
    assert row["terrain_build_ms"] == 1.0
    assert row["plan_ms"] == 5.0
    assert row["playback_ms"] == 6.0
    assert row["qp_nominal_ms"] == 1.25
    assert row["qp_solve_ms"] == 2.5
    assert row["qp_total_ms"] == 4.5


def test_mpc_qp_viewer_zero_command_hold_uses_previous_result_final_frame() -> None:
    from extension.viz import go2_foostep_planner as viewer

    horizon = 6
    root = torch.zeros((1, horizon, 3), dtype=torch.float64)
    root[0, :, 0] = torch.arange(horizon, dtype=torch.float64)
    quat = torch.zeros((1, horizon, 4), dtype=torch.float64)
    quat[..., 0] = 1.0
    joint = torch.arange(horizon * 12, dtype=torch.float64).reshape(1, horizon, 12)
    foot = torch.arange(horizon * 4 * 3, dtype=torch.float64).reshape(1, horizon, 4, 3)
    touchdown = foot + 0.5
    result = viewer.ViewerTrajectoryResult(
        num_frames=horizon,
        root_pos_w=root,
        root_quat_w=quat,
        joint_angles=joint,
        foot_pos_w=foot,
        foot_pos_root=foot - root[:, :, None, :],
        contact_state=torch.zeros((1, horizon, 4), dtype=torch.bool),
        planned_touchdown_w=touchdown,
        touchdown_seq=touchdown[:, 0].transpose(0, 1).unsqueeze(0),
    )

    held = viewer._viewer_hold_result_from_previous_final_frame(result)

    assert held.num_frames == horizon
    torch.testing.assert_close(held.root_pos_w, root[:, -1:].expand_as(root))
    torch.testing.assert_close(held.joint_angles, joint[:, -1:].expand_as(joint))
    torch.testing.assert_close(held.foot_pos_w, foot[:, -1:].expand_as(foot))
    torch.testing.assert_close(held.planned_touchdown_w, touchdown[:, -1:].expand_as(touchdown))
    assert torch.equal(held.contact_state, torch.ones_like(held.contact_state, dtype=torch.bool))


def test_mpc_qp_viewer_zero_command_forces_hold_when_previous_result_still_moves() -> None:
    from extension.viz import go2_foostep_planner as viewer

    horizon = 4
    root = torch.zeros((1, horizon, 3), dtype=torch.float64)
    root[0, :, 0] = torch.arange(horizon, dtype=torch.float64)
    quat = torch.zeros((1, horizon, 4), dtype=torch.float64)
    quat[..., 0] = 1.0
    result = viewer.ViewerTrajectoryResult(
        num_frames=horizon,
        root_pos_w=root,
        root_quat_w=quat,
        joint_angles=torch.zeros((1, horizon, 12), dtype=torch.float64),
        foot_pos_w=torch.zeros((1, horizon, 4, 3), dtype=torch.float64),
        foot_pos_root=torch.zeros((1, horizon, 4, 3), dtype=torch.float64),
        contact_state=torch.zeros((1, horizon, 4), dtype=torch.bool),
        planned_touchdown_w=torch.zeros((1, horizon, 4, 3), dtype=torch.float64),
    )

    assert viewer._viewer_should_hold_mpc_qp_zero_command(
        backend="mpc_qp",
        result=result,
        command=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert not viewer._viewer_should_hold_mpc_qp_zero_command(
        backend="mpc",
        result=result,
        command=torch.zeros((1, 3), dtype=torch.float64),
    )
    assert not viewer._viewer_should_hold_mpc_qp_zero_command(
        backend="mpc_qp",
        result=result,
        command=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64),
    )


def test_mpc_qp_viewer_zero_command_drains_active_motion_before_holding() -> None:
    from extension.viz import go2_foostep_planner as viewer

    horizon = 4
    root = torch.zeros((1, horizon, 3), dtype=torch.float64)
    root[0, :, 0] = torch.arange(horizon, dtype=torch.float64)
    result = SimpleNamespace(num_frames=horizon, root_pos_w=root)
    zero = torch.zeros((1, 3), dtype=torch.float64)
    previous_nonzero = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)

    assert viewer._viewer_should_drain_before_zero_replan(
        backend="mpc_qp",
        result=result,
        playback_frame=1,
        teleop_values=zero,
        last_cmd=previous_nonzero,
    )
    assert not viewer._viewer_should_hold_mpc_qp_zero_command(
        backend="mpc_qp",
        result=result,
        command=zero,
        drain_current_trajectory=True,
    )


def test_mpc_qp_viewer_zero_command_without_result_uses_static_hold_path() -> None:
    from extension.viz import go2_foostep_planner as viewer

    zero = torch.zeros((1, 3), dtype=torch.float64)

    assert viewer._viewer_should_static_hold_mpc_qp_zero_command(
        backend="mpc_qp",
        result=None,
        command=zero,
        drain_current_trajectory=False,
    )
    assert not viewer._viewer_should_static_hold_mpc_qp_zero_command(
        backend="mpc",
        result=None,
        command=zero,
        drain_current_trajectory=False,
    )
    assert not viewer._viewer_should_static_hold_mpc_qp_zero_command(
        backend="mpc_qp",
        result=None,
        command=torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64),
        drain_current_trajectory=False,
    )


def test_mpc_qp_viewer_static_hold_result_from_current_state_is_stationary() -> None:
    from extension.viz import go2_foostep_planner as viewer

    state = SimpleNamespace(
        root_pos=torch.tensor([[1.0, 2.0, 0.35]], dtype=torch.float64),
        root_rpy=torch.tensor([[0.0, 0.0, 0.25]], dtype=torch.float64),
        joint_angles=torch.arange(12, dtype=torch.float64).reshape(1, 12),
        foot_pos=torch.arange(12, dtype=torch.float64).reshape(1, 4, 3) * 0.01,
    )

    held = viewer._viewer_static_hold_result_from_current_state(state, horizon=5)

    assert held.num_frames == 5
    torch.testing.assert_close(held.root_pos_w, state.root_pos[:, None, :].expand(1, 5, 3))
    torch.testing.assert_close(held.joint_angles, state.joint_angles[:, None, :].expand(1, 5, 12))
    torch.testing.assert_close(held.foot_pos_w, state.foot_pos[:, None, :, :].expand(1, 5, 4, 3))
    torch.testing.assert_close(held.planned_touchdown_w, held.foot_pos_w)
    torch.testing.assert_close(held.foot_pos_root, held.foot_pos_w - held.root_pos_w[:, :, None, :])
    assert torch.equal(held.contact_state, torch.ones((1, 5, 4), dtype=torch.bool))
    assert torch.count_nonzero(held.root_lin_vel_w) == 0
    assert torch.count_nonzero(held.root_ang_vel_w) == 0


def test_mpc_qp_viewer_cli_overrides_sync_runtime_to_qp_cfg(monkeypatch) -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.viz import go2_foostep_planner as viewer

    fake_isaaclab = ModuleType("isaaclab")
    fake_app = ModuleType("isaaclab.app")

    class _FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", type=str, default="cpu")

    fake_app.AppLauncher = _FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)

    parser = build_arg_parser()
    args = parser.parse_args(
        ["--planner-backend", "mpc_qp", "--n-frames", "37", "--plan-dt", "0.037", "--qp-iterations", "4"]
    )
    env_cfg = SimpleNamespace(
        planner_backend="mpc_qp",
        mpc_planner_cfg=SimpleNamespace(runtime=SimpleNamespace(horizon_steps=0, replan_interval_steps=0, dt=0.0)),
        mpc_qp_planner_cfg=MpcQpPlannerCfg(),
    )

    viewer._apply_planner_runtime_cli_overrides(env_cfg, args)

    assert env_cfg.mpc_qp_planner_cfg.runtime.horizon_steps == 37
    assert env_cfg.mpc_qp_planner_cfg.runtime.replan_interval_steps == 37
    assert env_cfg.mpc_qp_planner_cfg.runtime.dt == 0.037
    assert env_cfg.mpc_qp_planner_cfg.runtime.qp_iterations == 4


def test_mpc_qp_perf_probe_argparse_accepts_backend_and_iteration(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mpc_rl_epoch_perf_probe.py", "--planner-backend", "mpc_qp", "--qp-iterations", "2"],
    )

    args = _parse_perf_probe_args()

    assert args.planner_backend == "mpc_qp"
    assert args.qp_iterations == 2


def test_mpc_qp_perf_probe_records_historical_qp_iteration_metrics() -> None:
    source = (GO2PVCNN_ROOT / "tests/mpc_rl_epoch_perf_probe.py").read_text()

    assert "max_qp_iterations_executed_seen" in source
    assert "qp_replan_event_count" in source
    assert "max_qp_solve_ms_seen" in source
    assert "max_qp_total_ms_seen" in source


def test_mpc_qp_cubic_bezier_sampling_is_continuous_and_endpoint_exact() -> None:
    from extension.batch_mpc_qp_planner.bezier import (
        cubic_bezier_basis,
        sample_cubic_bezier,
        trajectory_frame_deltas,
    )

    controls = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.2, 0.0, 0.3], [0.4, 0.0, 0.3], [0.6, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    basis = cubic_bezier_basis(9, dtype=controls.dtype, device=controls.device)
    samples = sample_cubic_bezier(controls, basis)
    first_delta, second_delta = trajectory_frame_deltas(samples)

    assert samples.shape == (1, 1, 9, 3)
    assert torch.allclose(samples[:, :, 0], controls[:, :, 0])
    assert torch.allclose(samples[:, :, -1], controls[:, :, -1])
    assert torch.isfinite(first_delta).all()
    assert torch.isfinite(second_delta).all()
    assert float(first_delta.norm(dim=-1).amax().item()) < 0.20


def test_mpc_qp_continuous_decode_binds_touchdown_z_to_terrain() -> None:
    from extension.batch_mpc_planner.terrain import height_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.continuous import build_controls_from_nominal, decode_controls_to_result
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 1
    terrain = _flat_terrain(batch=1, size=9)
    terrain.height_map[:, 4:, :] = 0.11
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    nominal = plan_segment_qp(terrain, state, command, cfg=cfg)

    controls = build_controls_from_nominal(nominal, terrain)
    decoded = decode_controls_to_result(nominal, terrain, controls, sample_count=cfg.runtime.horizon_steps)
    touchdown_xy = decoded.planned_touchdown_w[:, 0, :, :2]
    expected_z = height_at(terrain, touchdown_xy).to(dtype=decoded.planned_touchdown_w.dtype)

    assert torch.allclose(decoded.planned_touchdown_w[:, 0, :, 2], expected_z, atol=1.0e-5)
    assert torch.allclose(decoded.touchdown_seq[:, :, 0, 2], expected_z, atol=1.0e-5)


def test_mpc_qp_continuous_loss_diagnostics_report_smoothness_and_foothold_quality() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.losses import continuous_loss_diagnostics
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=2, size=9)
    terrain.height_map[:, 4, :] = 0.08
    state = _state(batch=2)
    command = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.20, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    diag = continuous_loss_diagnostics(result, terrain, footprint_radius_m=0.04)

    for key in (
        "qp_continuous_foot_frame_jump_max",
        "qp_continuous_foot_acceleration_max",
        "qp_continuous_foothold_height_variation_max",
        "qp_continuous_touchdown_semantic_bad_count",
    ):
        assert key in diag
        assert diag[key].shape == (2,)
        assert torch.isfinite(diag[key]).all()


def test_mpc_qp_continuous_loss_diagnostics_measure_foot_jump_over_time_not_between_legs() -> None:
    from extension.batch_mpc_planner.types import MpcPlannerResult
    from extension.batch_mpc_qp_planner.losses import continuous_loss_diagnostics

    terrain = _flat_terrain(batch=1, size=9)
    foot = torch.tensor(
        [[
            [[0.20, 0.12, 0.0], [0.20, -0.12, 0.0], [-0.20, 0.12, 0.0], [-0.20, -0.12, 0.0]],
            [[0.20, 0.12, 0.0], [0.20, -0.12, 0.0], [-0.20, 0.12, 0.0], [-0.20, -0.12, 0.0]],
            [[0.20, 0.12, 0.0], [0.20, -0.12, 0.0], [-0.20, 0.12, 0.0], [-0.20, -0.12, 0.0]],
        ]],
        dtype=torch.float32,
    )
    result = MpcPlannerResult(
        root_pos=torch.zeros((1, 3, 3), dtype=torch.float32),
        root_rpy=torch.zeros((1, 3, 3), dtype=torch.float32),
        foot_pos=foot,
        joint_angles=torch.zeros((1, 3, 12), dtype=torch.float32),
        contact_state=torch.ones((1, 3, 4), dtype=torch.bool),
        touchdown_seq=foot[:, 0].unsqueeze(2),
        planned_touchdown_w=foot[:, :1],
        cost_total=torch.zeros(1, dtype=torch.float32),
        cost_breakdown={},
        status=torch.zeros(1, dtype=torch.long),
        feasible=torch.ones(1, dtype=torch.bool),
        safe_fallback=torch.zeros(1, dtype=torch.bool),
    )

    diag = continuous_loss_diagnostics(result, terrain, footprint_radius_m=0.04)

    assert diag["qp_continuous_foot_frame_jump_max"].item() == 0.0
    assert diag["qp_continuous_foot_acceleration_max"].item() == 0.0


def test_mpc_qp_continuous_loss_diagnostics_measure_joint_jump_per_joint_not_vector_norm() -> None:
    from extension.batch_mpc_planner.types import MpcPlannerResult
    from extension.batch_mpc_qp_planner.losses import continuous_loss_diagnostics

    terrain = _flat_terrain(batch=1, size=9)
    foot = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    joint = torch.zeros((1, 2, 12), dtype=torch.float32)
    joint[:, 1, :] = 0.20
    result = MpcPlannerResult(
        root_pos=torch.zeros((1, 2, 3), dtype=torch.float32),
        root_rpy=torch.zeros((1, 2, 3), dtype=torch.float32),
        foot_pos=foot,
        joint_angles=joint,
        contact_state=torch.ones((1, 2, 4), dtype=torch.bool),
        touchdown_seq=foot[:, 0].unsqueeze(2),
        planned_touchdown_w=foot[:, :1],
        cost_total=torch.zeros(1, dtype=torch.float32),
        cost_breakdown={},
        status=torch.zeros(1, dtype=torch.long),
        feasible=torch.ones(1, dtype=torch.bool),
        safe_fallback=torch.zeros(1, dtype=torch.bool),
    )

    diag = continuous_loss_diagnostics(result, terrain, footprint_radius_m=0.04)

    assert torch.allclose(diag["qp_continuous_joint_frame_jump_max"], torch.tensor([0.20]))


def test_mpc_qp_differentiable_fields_return_fixed_shape_values_and_gradients() -> None:
    from extension.batch_mpc_qp_planner.fields import build_qp_fields

    terrain = _flat_terrain(batch=2, size=11)
    terrain.height_map[:, 5:, :] = 0.10
    terrain.semantic_map[:, :, 6:] = 1
    fields = build_qp_fields(terrain)
    query_xy = torch.tensor(
        [
            [[-0.10, -0.10], [0.05, 0.05], [0.25, 0.25]],
            [[-0.20, 0.15], [0.00, 0.00], [0.30, -0.10]],
        ],
        dtype=torch.float32,
    )

    sample = fields.query(query_xy)

    assert sample.height.shape == (2, 3)
    assert sample.height_grad_xy.shape == (2, 3, 2)
    assert sample.semantic_risk.shape == (2, 3)
    assert sample.semantic_grad_xy.shape == (2, 3, 2)
    assert sample.roughness.shape == (2, 3)
    assert sample.roughness_grad_xy.shape == (2, 3, 2)
    assert torch.isfinite(sample.height).all()
    assert torch.isfinite(sample.height_grad_xy).all()
    assert torch.isfinite(sample.semantic_risk).all()
    assert torch.isfinite(sample.semantic_grad_xy).all()
    assert torch.isfinite(sample.roughness).all()
    assert torch.isfinite(sample.roughness_grad_xy).all()


def test_mpc_qp_gait_masks_encode_fixed_alternating_diagonal_phases() -> None:
    from extension.batch_mpc_qp_planner.gait import alternating_diagonal_gait_masks

    gait = alternating_diagonal_gait_masks(batch=2, horizon=8, device=torch.device("cpu"))

    assert gait.swing_mask.shape == (2, 8, 4)
    assert gait.stance_mask.shape == (2, 8, 4)
    assert torch.equal(gait.swing_mask[:, :4, 1], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.swing_mask[:, :4, 2], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.stance_mask[:, :4, 0], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.stance_mask[:, :4, 3], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.swing_mask[:, 4:, 0], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.swing_mask[:, 4:, 3], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.stance_mask[:, 4:, 1], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.stance_mask[:, 4:, 2], torch.ones((2, 4), dtype=torch.bool))
    assert torch.equal(gait.swing_mask, torch.logical_not(gait.stance_mask))


def test_mpc_qp_gait_masks_use_all_stance_for_zero_command_idle() -> None:
    from extension.batch_mpc_qp_planner.gait import alternating_diagonal_gait_masks

    command = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float32)
    gait = alternating_diagonal_gait_masks(
        batch=2,
        horizon=8,
        device=torch.device("cpu"),
        command=command,
        idle_command_threshold=1.0e-3,
    )

    assert torch.count_nonzero(gait.swing_mask[0]) == 0
    assert torch.count_nonzero(torch.logical_not(gait.stance_mask[0])) == 0
    assert torch.count_nonzero(gait.swing_mask[1]) > 0


def test_mpc_qp_plan_segment_outputs_fixed_alternating_diagonal_contact_state() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.gait import alternating_diagonal_gait_masks
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=2, size=9)
    state = _state(batch=2)
    command = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.20, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    expected = alternating_diagonal_gait_masks(
        batch=2,
        horizon=cfg.runtime.horizon_steps,
        device=result.contact_state.device,
    ).stance_mask

    assert torch.equal(result.contact_state, expected)
    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_fixed_gait_active"], torch.ones(2))


def test_mpc_qp_zero_command_idle_anchors_joint_state_and_limits_jumps() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 1
    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.zeros((1, 3), dtype=torch.float32)

    first = plan_segment_qp(terrain, state, command, cfg=cfg)
    next_state = MpcRobotState(
        root_pos=first.root_pos[:, -1],
        root_rpy=first.root_rpy[:, -1],
        foot_pos=first.foot_pos[:, -1],
        joint_angles=first.joint_angles[:, -1],
    )
    second = plan_segment_qp(terrain, next_state, command, cfg=cfg)
    joint_delta = torch.abs(second.joint_angles[:, 1:] - second.joint_angles[:, :-1]).amax()

    assert torch.equal(second.contact_state, torch.ones_like(second.contact_state, dtype=torch.bool))
    torch.testing.assert_close(second.joint_angles[:, 0], next_state.joint_angles, atol=1.0e-6, rtol=0.0)
    assert float(joint_delta.item()) <= 0.04
    assert second.loss_breakdown is not None
    assert torch.equal(second.loss_breakdown["qp_idle_all_stance_active"], torch.ones(1))


def test_mpc_qp_manager_holds_previous_plan_final_frame_for_zero_command_rows() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.manager import MpcQpTrajectoryManager
    from extension.reference.cache import ReferenceTrajectoryCache

    cfg = SimpleNamespace(mpc_qp_planner_cfg=MpcQpPlannerCfg())
    manager = MpcQpTrajectoryManager(cfg, device="cpu")
    horizon = int(cfg.mpc_qp_planner_cfg.runtime.horizon_steps)
    root = torch.zeros((2, horizon, 3), dtype=torch.float32)
    root[0, :, 0] = torch.arange(horizon, dtype=torch.float32)
    root[1, :, 0] = 100.0 + torch.arange(horizon, dtype=torch.float32)
    quat = torch.zeros((2, horizon, 4), dtype=torch.float32)
    quat[..., 0] = 1.0
    joint = torch.arange(2 * horizon * 12, dtype=torch.float32).reshape(2, horizon, 12)
    foot = torch.arange(2 * horizon * 4 * 3, dtype=torch.float32).reshape(2, horizon, 4, 3)
    manager._cache = ReferenceTrajectoryCache(
        root_pos_w=root.clone(),
        root_quat_w=quat.clone(),
        joint_angles=joint.clone(),
        foot_pos_w=foot.clone(),
        foot_pos_root=foot.clone() - root[:, :, None, :],
        contact_state=torch.zeros((2, horizon, 4), dtype=torch.bool),
        planned_touchdown_w=foot.clone(),
        phase_index=torch.arange(horizon, dtype=torch.long).unsqueeze(0).expand(2, horizon).clone(),
        valid_mask=torch.ones((2, horizon), dtype=torch.bool),
    )

    selected_ids = torch.tensor([0, 1], dtype=torch.long)
    command = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float32)
    plan_ids, hold_ids = manager._split_selected_ids_for_planning(selected_ids, command, cache_valid=True)

    assert torch.equal(plan_ids, torch.tensor([1], dtype=torch.long))
    assert torch.equal(hold_ids, torch.tensor([0], dtype=torch.long))
    manager._apply_hold_cache_rows_from_previous_final(hold_ids)

    assert manager._cache is not None
    torch.testing.assert_close(manager._cache.root_pos_w[0], root[0, -1:].expand(horizon, 3))
    torch.testing.assert_close(manager._cache.joint_angles[0], joint[0, -1:].expand(horizon, 12))
    torch.testing.assert_close(manager._cache.foot_pos_w[0], foot[0, -1:].expand(horizon, 4, 3))
    torch.testing.assert_close(manager._cache.root_pos_w[1], root[1])


def test_mpc_qp_fixed_gait_keeps_stance_feet_anchored() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    contact = result.contact_state[0]
    foot = result.foot_pos[0]
    split = cfg.runtime.horizon_steps // 2

    # FR/RL swing first, then hold touchdown as stance anchor.
    assert torch.allclose(foot[split:, 1], foot[split : split + 1, 1].expand_as(foot[split:, 1]), atol=1.0e-5)
    assert torch.allclose(foot[split:, 2], foot[split : split + 1, 2].expand_as(foot[split:, 2]), atol=1.0e-5)
    # FL/RR hold start anchor first, then swing.
    assert torch.allclose(foot[:split, 0], foot[:1, 0].expand_as(foot[:split, 0]), atol=1.0e-5)
    assert torch.allclose(foot[:split, 3], foot[:1, 3].expand_as(foot[:split, 3]), atol=1.0e-5)
    assert torch.count_nonzero(torch.logical_not(contact[:, 1])) == split
    assert torch.count_nonzero(torch.logical_not(contact[:, 0])) == cfg.runtime.horizon_steps - split


def test_mpc_qp_fixed_gait_places_touchdown_at_swing_to_stance_boundary() -> None:
    from extension.batch_mpc_planner.terrain import height_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    split = cfg.runtime.horizon_steps // 2
    touchdown = result.planned_touchdown_w[0, 0]

    assert torch.allclose(result.foot_pos[0, split, 1], touchdown[1], atol=1.0e-5)
    assert torch.allclose(result.foot_pos[0, split, 2], touchdown[2], atol=1.0e-5)
    assert torch.allclose(result.foot_pos[0, -1, 0], touchdown[0], atol=1.0e-5)
    assert torch.allclose(result.foot_pos[0, -1, 3], touchdown[3], atol=1.0e-5)
    start_anchor = state.foot_pos[0].clone()
    start_anchor[:, 2] = height_at(terrain, state.foot_pos[:, :, :2])[0]
    assert torch.allclose(result.foot_pos[0, :split, 0], start_anchor[0].expand_as(result.foot_pos[0, :split, 0]), atol=1.0e-5)
    assert torch.allclose(result.foot_pos[0, :split, 3], start_anchor[3].expand_as(result.foot_pos[0, :split, 3]), atol=1.0e-5)


def test_mpc_qp_fixed_gait_stance_anchors_bind_start_feet_to_terrain() -> None:
    from extension.batch_mpc_planner.terrain import height_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    state.foot_pos[..., 2] = 0.32
    command = torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    split = cfg.runtime.horizon_steps // 2
    expected_start_z = height_at(terrain, state.foot_pos[..., :2]).to(dtype=result.foot_pos.dtype)

    assert torch.allclose(result.foot_pos[0, :split, 0, 2], expected_start_z[0, 0].expand(split), atol=1.0e-5)
    assert torch.allclose(result.foot_pos[0, :split, 3, 2], expected_start_z[0, 3].expand(split), atol=1.0e-5)


def test_mpc_qp_variable_layout_contains_touchdown_xy_and_slacks() -> None:
    from extension.batch_mpc_qp_planner.variables import build_qp_variable_layout

    layout = build_qp_variable_layout(batch=2, horizon=6, legs=4, device=torch.device("cpu"))

    assert layout.total_dim > 0
    assert layout.touchdown_xy.shape == (2, 4, 2)
    assert layout.semantic_slack.shape == (2, 4)
    assert layout.clearance_slack.shape == (2, 4)
    assert layout.reachability_slack.shape == (2, 4)
    assert layout.stability_slack.shape == (2, 1)
    assert layout.touchdown_xy_indices.shape == (2, 4, 2)
    assert layout.touchdown_z_is_height_bound is True


def test_mpc_qp_assembly_returns_fixed_shape_qp_matrices() -> None:
    from extension.batch_mpc_qp_planner.fields import build_qp_fields
    from extension.batch_mpc_qp_planner.gait import alternating_diagonal_gait_masks
    from extension.batch_mpc_qp_planner.qp_assembly import assemble_fixed_shape_qp
    from extension.batch_mpc_qp_planner.variables import build_qp_variable_layout

    terrain = _flat_terrain(batch=2, size=11)
    fields = build_qp_fields(terrain)
    layout = build_qp_variable_layout(batch=2, horizon=6, legs=4, device=torch.device("cpu"))
    gait = alternating_diagonal_gait_masks(batch=2, horizon=6, device=torch.device("cpu"))
    qps = assemble_fixed_shape_qp(fields=fields, layout=layout, gait=gait)

    n = layout.total_dim
    assert qps.H.shape == (2, n, n)
    assert qps.g.shape == (2, n)
    assert qps.A.shape[0] == 2 and qps.A.shape[2] == n
    assert qps.b.shape == qps.A.shape[:2]
    assert qps.E.shape[0] == 2 and qps.E.shape[2] == n
    assert qps.e.shape == qps.E.shape[:2]
    assert qps.lower.shape == (2, n)
    assert qps.upper.shape == (2, n)
    assert torch.isfinite(qps.H).all()
    assert torch.isfinite(qps.g).all()


def test_mpc_qp_continuous_solver_source_has_no_candidate_or_search_main_path() -> None:
    source = (GO2PVCNN_ROOT / "extension/batch_mpc_qp_planner/solver.py").read_text()

    forbidden = (
        "_repeat_terrain_for_candidates",
        "candidate_xy",
        "best_idx",
        "best_score",
        "scales = torch.tensor",
        "best_scale_idx",
        "semantic_repair",
        "fixed_repair_offsets",
    )
    for token in forbidden:
        assert token not in source


def test_mpc_qp_plan_segment_uses_continuous_path_without_repair_main_path() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_continuous_enabled"], torch.ones(1))
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert "qp_continuous_foot_frame_jump_max" in result.loss_breakdown
    assert "qp_continuous_foothold_height_variation_max" in result.loss_breakdown


def test_mpc_qp_continuous_iterations_reduce_bad_foothold_height_variation_without_repair() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    probe_cfg = MpcQpPlannerCfg()
    probe_cfg.runtime.qp_iterations = 1
    probe = plan_segment_qp(terrain, state, command, cfg=probe_cfg)
    touchdown_x = float(probe.planned_touchdown_w[0, 0, 0, 0].item())
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    ix = int(torch.argmin(torch.abs(xs - touchdown_x)).item())
    terrain.height_map[:, :, ix:] = 0.12

    one_iter_cfg = MpcQpPlannerCfg()
    one_iter_cfg.runtime.qp_iterations = 1
    one_iter_cfg.runtime.continuous_foothold_variation_target_m = 0.0
    many_iter_cfg = MpcQpPlannerCfg()
    many_iter_cfg.runtime.qp_iterations = 4
    many_iter_cfg.runtime.continuous_foothold_variation_target_m = 0.0

    one_iter = plan_segment_qp(terrain, state, command, cfg=one_iter_cfg)
    many_iter = plan_segment_qp(terrain, state, command, cfg=many_iter_cfg)

    assert one_iter.loss_breakdown is not None
    assert many_iter.loss_breakdown is not None
    assert torch.equal(many_iter.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert many_iter.loss_breakdown["qp_continuous_foothold_height_variation_max"].item() < (
        one_iter.loss_breakdown["qp_continuous_foothold_height_variation_max"].item()
    )
    assert many_iter.loss_breakdown["qp_continuous_solver_update_count"].item() > 0


def test_mpc_qp_continuous_iterations_reduce_touchdown_semantic_bad_without_repair() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    probe = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())
    touchdown_xy = probe.planned_touchdown_w[0, 0, 0, :2].detach()
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - touchdown_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - touchdown_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1

    one_iter_cfg = MpcQpPlannerCfg()
    one_iter_cfg.runtime.qp_iterations = 1
    many_iter_cfg = MpcQpPlannerCfg()
    many_iter_cfg.runtime.qp_iterations = 6

    one_iter = plan_segment_qp(terrain, state, command, cfg=one_iter_cfg)
    many_iter = plan_segment_qp(terrain, state, command, cfg=many_iter_cfg)

    assert one_iter.loss_breakdown is not None
    assert many_iter.loss_breakdown is not None
    assert torch.equal(many_iter.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert many_iter.loss_breakdown["qp_continuous_touchdown_semantic_bad_count"].item() < (
        one_iter.loss_breakdown["qp_continuous_touchdown_semantic_bad_count"].item()
    )
    assert many_iter.loss_breakdown["qp_continuous_touchdown_semantic_bad_count"].item() == 0
    assert many_iter.loss_breakdown["qp_continuous_solver_update_count"].item() > 0
    assert many_iter.loss_breakdown["qp_continuous_solver_semantic_score_before_max"].item() > 0
    assert many_iter.loss_breakdown["qp_continuous_solver_semantic_score_after_max"].item() == 0


def test_mpc_qp_continuous_iterations_lift_swing_over_low_small_without_repair() -> None:
    from extension.batch_mpc_planner.terrain import height_at, semantic_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    probe = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())
    swing_idx = torch.nonzero(torch.logical_not(probe.contact_state[0, :, 0]), as_tuple=False).flatten()
    obstacle_xy = probe.foot_pos[0, swing_idx[len(swing_idx) // 2], 0, :2].detach()
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - obstacle_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - obstacle_xy[1])).item())
    terrain.height_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 0.08
    terrain.semantic_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 1

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 3
    cfg.runtime.low_small_swing_clearance_m = 0.18
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    foot_semantic = semantic_at(terrain, result.foot_pos[..., :2])
    contact = result.contact_state
    prev_contact = torch.nn.functional.pad(contact[:, :-1], (0, 0, 1, 0), value=True)
    next_contact = torch.nn.functional.pad(contact[:, 1:], (0, 0, 0, 1), value=True)
    mid_swing = torch.logical_and(torch.logical_not(contact), torch.logical_and(torch.logical_not(prev_contact), torch.logical_not(next_contact)))
    low_small_swing = torch.logical_and(
        torch.logical_and(foot_semantic == 1, torch.logical_not(result.contact_state)),
        mid_swing,
    )
    assert torch.count_nonzero(low_small_swing) > 0
    terrain_z = height_at(terrain, result.foot_pos[..., :2]).to(dtype=result.foot_pos.dtype)
    clearance = result.foot_pos[..., 2] - terrain_z
    assert torch.max(clearance[low_small_swing]).item() >= cfg.runtime.low_small_swing_clearance_m - 1.0e-5
    assert result.loss_breakdown["qp_continuous_low_small_clearance_deficit_max"].item() >= 0.0
    assert result.loss_breakdown["qp_continuous_solver_swing_clearance_lift_count"].item() >= 0


def test_mpc_qp_continuous_reports_terrain_clearance_and_swing_height_metrics() -> None:
    from extension.batch_mpc_planner.terrain import height_at, semantic_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    probe = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())
    swing_idx = torch.nonzero(torch.logical_not(probe.contact_state[0, :, 0]), as_tuple=False).flatten()
    obstacle_xy = probe.foot_pos[0, swing_idx[len(swing_idx) // 2], 0, :2].detach()
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - obstacle_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - obstacle_xy[1])).item())
    terrain.height_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 0.05
    terrain.semantic_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 1

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 2
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    for key in (
        "qp_continuous_planned_foot_terrain_clearance_min",
        "qp_continuous_fk_foot_terrain_clearance_min",
        "qp_continuous_planned_foot_terrain_penetration_count",
        "qp_continuous_fk_foot_terrain_penetration_count",
        "qp_continuous_swing_height_over_terrain_max",
        "qp_continuous_low_small_swing_height_over_terrain_max",
    ):
        assert key in result.loss_breakdown

    planned_terrain_z = height_at(terrain, result.foot_pos[..., :2]).to(dtype=result.foot_pos.dtype)
    planned_clearance = result.foot_pos[..., 2] - planned_terrain_z
    assert torch.allclose(
        result.loss_breakdown["qp_continuous_planned_foot_terrain_clearance_min"],
        planned_clearance.reshape(1, -1).amin(dim=1),
        atol=1.0e-6,
    )
    assert result.loss_breakdown["qp_continuous_planned_foot_terrain_penetration_count"].item() == 0
    assert result.loss_breakdown["qp_continuous_fk_foot_terrain_penetration_count"].item() == 0

    foot_semantic = semantic_at(terrain, result.foot_pos[..., :2])
    swing = torch.logical_not(result.contact_state)
    low_small_swing = torch.logical_and(foot_semantic == 1, swing)
    assert torch.count_nonzero(low_small_swing) > 0
    assert result.loss_breakdown["qp_continuous_low_small_swing_height_over_terrain_max"].item() <= 0.17


def test_mpc_qp_continuous_reports_fk_readback_error_without_repair() -> None:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 2
    terrain = _flat_terrain(batch=2, size=17)
    state = _state(batch=2)
    state.foot_pos[..., 2] = 0.0
    command = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.20, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    readback_error = torch.linalg.vector_norm(fk_foot - result.foot_pos, dim=-1).reshape(2, -1).amax(dim=1)
    assert "qp_continuous_fk_readback_error_max" in result.loss_breakdown
    assert "qp_continuous_fk_readback_error_mean" in result.loss_breakdown
    assert torch.allclose(result.loss_breakdown["qp_continuous_fk_readback_error_max"], readback_error, atol=1.0e-6)
    assert torch.count_nonzero(result.loss_breakdown["qp_continuous_fk_readback_error_max"] > 0.035) == 0
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(2))


def test_mpc_qp_continuous_root_progress_reduces_on_height_edge_without_repair() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    state.foot_pos[..., 2] = 0.0
    command = torch.tensor([[0.55, 0.0, 0.0]], dtype=torch.float32)
    probe = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())
    root_end_x = float(probe.root_pos[0, -1, 0].item())
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    ix = int(torch.argmin(torch.abs(xs - (root_end_x * 0.55))).item())
    terrain.height_map[:, :, ix:] = 0.12

    one_iter_cfg = MpcQpPlannerCfg()
    one_iter_cfg.runtime.qp_iterations = 1
    many_iter_cfg = MpcQpPlannerCfg()
    many_iter_cfg.runtime.qp_iterations = 4

    one_iter = plan_segment_qp(terrain, state, command, cfg=one_iter_cfg)
    many_iter = plan_segment_qp(terrain, state, command, cfg=many_iter_cfg)

    assert one_iter.loss_breakdown is not None
    assert many_iter.loss_breakdown is not None
    assert torch.equal(many_iter.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert one_iter.root_pos[0, -1, 0].item() < probe.root_pos[0, -1, 0].item()
    assert many_iter.root_pos[0, -1, 0].item() <= one_iter.root_pos[0, -1, 0].item() + 1.0e-6
    assert many_iter.loss_breakdown["qp_continuous_root_terrain_risk_reduces_progress"].item() > 0
    assert many_iter.loss_breakdown["qp_continuous_root_progress_scale_min"].item() < 1.0


def test_mpc_qp_continuous_root_progress_does_not_cap_over_low_small_obstacle() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=81)
    state = _state(batch=1)
    state.foot_pos[..., 2] = 0.0
    command = torch.tensor([[0.45, 0.0, 0.0]], dtype=torch.float32)
    probe = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())
    root_mid_x = float(probe.root_pos[0, probe.root_pos.shape[1] // 2, 0].item())
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.height_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - root_mid_x)).item())
    iy = int(torch.argmin(torch.abs(ys - 0.0)).item())
    terrain.height_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 0.16
    terrain.semantic_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 1

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 2
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert result.loss_breakdown["qp_continuous_root_terrain_risk_reduces_progress"].item() == 0
    assert result.loss_breakdown["qp_continuous_root_progress_scale_min"].item() == 1.0
    assert result.root_pos[0, -1, 0].item() >= probe.root_pos[0, -1, 0].item() - 1.0e-5


def test_mpc_qp_continuous_low_small_crossing_progress_reaches_obstacle_window_without_forcing_foot_over() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=81)
    state = _state(batch=1)
    state.root_pos[0, 0] = -0.35
    state.foot_pos[0, :, 0] += -0.35
    state.foot_pos[..., 2] = 0.0
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.height_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - 0.0)).item())
    iy = int(torch.argmin(torch.abs(ys - 0.0)).item())
    terrain.height_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 0.05
    terrain.semantic_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 1

    cfg = MpcQpPlannerCfg()
    cfg.runtime.horizon_steps = 50
    cfg.runtime.qp_iterations = 2
    result = plan_segment_qp(terrain, state, torch.tensor([[0.45, 0.0, 0.0]], dtype=torch.float32), cfg=cfg)

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_repair_main_path_active"].item() == 0
    assert result.root_pos[0, 0, 0].item() < -0.30
    assert result.root_pos[0, -1, 0].item() > 0.02
    assert result.loss_breakdown["qp_continuous_low_small_progress_update_count"].item() > 0
    assert result.loss_breakdown["qp_continuous_low_small_foot_over_update_count"].item() == 0
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_touchdown_on_small_count"].item() == 0
    assert result.loss_breakdown["qp_continuous_planned_foot_terrain_penetration_count"].item() == 0


def test_mpc_qp_continuous_low_small_qp_creates_crossing_leg_from_trajectory_loss() -> None:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp
    from mpc_low_small_reachable_crossing_probe import reachable_foot_over_arc_metrics

    terrain = _flat_terrain(batch=1, size=81)
    state = _state(batch=1)
    state.root_pos[0, 0] = -0.35
    state.foot_pos[0, :, 0] += -0.35
    state.foot_pos[..., 2] = 0.0
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.height_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - 0.0)).item())
    iy = int(torch.argmin(torch.abs(ys - 0.0)).item())
    terrain.height_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 0.05
    terrain.semantic_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 1
    obstacle_xy = torch.stack((xs[ix], ys[iy]))

    cfg = MpcQpPlannerCfg()
    cfg.runtime.horizon_steps = 50
    cfg.runtime.qp_iterations = 2
    result = plan_segment_qp(terrain, state, torch.tensor([[0.45, 0.0, 0.0]], dtype=torch.float32), cfg=cfg)
    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    metrics = reachable_foot_over_arc_metrics(
        fk_foot,
        result.contact_state,
        obstacle_xy,
        command=(0.45, 0.0, 0.0),
        obstacle_height=0.05,
        clearance=0.05,
        lane_half_width=0.14,
    )

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_repair_main_path_active"].item() == 0
    assert result.loss_breakdown["qp_continuous_low_small_crossing_leg_count"].item() > 0
    assert metrics["fk_foot_over_low_small_success"] == 1
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_touchdown_on_small_count"].item() == 0
    assert result.loss_breakdown["qp_continuous_fk_readback_error_max"].item() <= 0.05


def test_mpc_qp_continuous_low_small_high_arc_remains_fk_reachable() -> None:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp
    from mpc_low_small_reachable_crossing_probe import reachable_foot_over_arc_metrics

    terrain = _flat_terrain(batch=1, size=81)
    state = _state(batch=1)
    state.root_pos[0, 0] = -0.35
    state.foot_pos[0, :, 0] += -0.35
    state.foot_pos[..., 2] = 0.0
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.height_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - 0.0)).item())
    iy = int(torch.argmin(torch.abs(ys - 0.0)).item())
    terrain.height_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 0.16
    terrain.semantic_map[:, max(0, iy - 2) : iy + 3, max(0, ix - 2) : ix + 3] = 1
    obstacle_xy = torch.stack((xs[ix], ys[iy]))

    cfg = MpcQpPlannerCfg()
    cfg.runtime.horizon_steps = 50
    cfg.runtime.qp_iterations = 2
    result = plan_segment_qp(terrain, state, torch.tensor([[0.45, 0.0, 0.0]], dtype=torch.float32), cfg=cfg)
    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    metrics = reachable_foot_over_arc_metrics(
        fk_foot,
        result.contact_state,
        obstacle_xy,
        command=(0.45, 0.0, 0.0),
        obstacle_height=0.16,
        clearance=0.05,
        lane_half_width=0.06,
    )

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_repair_main_path_active"].item() == 0
    assert result.loss_breakdown["qp_touchdown_on_small_count"].item() == 0
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_continuous_low_small_crossing_leg_count"].item() > 0
    assert metrics["fk_foot_over_low_small_success"] == 1
    assert result.loss_breakdown["qp_continuous_fk_readback_error_max"].item() <= 0.05
    assert result.loss_breakdown["qp_fk_body_leg_collision_count"].item() == 0
    assert result.loss_breakdown["qp_continuous_low_small_swing_height_over_terrain_max"].item() <= 0.18


def test_mpc_qp_continuous_root_progress_cap_keeps_foot_controls_reachable_without_repair() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    state.foot_pos[..., 2] = 0.0
    command = torch.tensor([[0.55, 0.0, 0.0]], dtype=torch.float32)
    probe = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())
    root_end_x = float(probe.root_pos[0, -1, 0].item())
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    ix = int(torch.argmin(torch.abs(xs - (root_end_x * 0.55))).item())
    terrain.height_map[:, :, ix:] = 0.22

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 1
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert result.loss_breakdown["qp_continuous_root_terrain_risk_reduces_progress"].item() > 0
    assert result.root_pos[0, -1, 0].item() < probe.root_pos[0, -1, 0].item()
    assert result.loss_breakdown["qp_continuous_fk_readback_error_max"].item() <= 0.05


def test_mpc_qp_continuous_height_edge_keeps_planned_and_fk_feet_above_terrain() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=81)
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    edge_ix = int(torch.argmin(torch.abs(xs - 0.08)).item())
    terrain.height_map[:, :, edge_ix:] = 0.18
    terrain.is_plane_terrain[:] = False
    state = _state(batch=1)
    state.foot_pos[..., 2] = 0.0
    command = torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32)

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 2
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert result.loss_breakdown["qp_continuous_planned_foot_terrain_penetration_count"].item() == 0
    assert result.loss_breakdown["qp_continuous_fk_foot_terrain_penetration_count"].item() == 0
    assert result.loss_breakdown["qp_continuous_planned_foot_terrain_clearance_min"].item() >= -1.0e-4
    assert result.loss_breakdown["qp_continuous_fk_foot_terrain_clearance_min"].item() >= -1.0e-4


def test_mpc_qp_continuous_fk_readback_update_improves_with_qp_iterations() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    edge_ix = int(torch.argmin(torch.abs(xs - 0.10)).item())
    terrain.height_map[:, :, edge_ix:] = 0.22
    state = _state(batch=1)
    state.foot_pos[..., 2] = 0.0
    command = torch.tensor([[0.55, 0.0, 0.0]], dtype=torch.float32)

    one_iter_cfg = MpcQpPlannerCfg()
    one_iter_cfg.runtime.qp_iterations = 1
    two_iter_cfg = MpcQpPlannerCfg()
    two_iter_cfg.runtime.qp_iterations = 2
    one_iter = plan_segment_qp(terrain, state, command, cfg=one_iter_cfg)
    two_iter = plan_segment_qp(terrain, state, command, cfg=two_iter_cfg)

    assert one_iter.loss_breakdown is not None
    assert two_iter.loss_breakdown is not None
    assert torch.equal(one_iter.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert torch.equal(two_iter.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert one_iter.loss_breakdown["qp_continuous_solver_fk_endpoint_update_count"].item() > 0
    assert two_iter.loss_breakdown["qp_continuous_fk_readback_error_max"].item() < (
        one_iter.loss_breakdown["qp_continuous_fk_readback_error_max"].item()
    )
    assert two_iter.loss_breakdown["qp_continuous_fk_readback_error_max"].item() <= 0.05


def test_mpc_qp_continuous_reachability_shortens_unreachable_low_touchdown_without_repair() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=81)
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.height_map.shape[-1])
    edge_ix = int(torch.argmin(torch.abs(xs - 0.30)).item())
    terrain.height_map[:, :, :edge_ix] = 0.70
    terrain.height_map[:, :, edge_ix:] = 0.0
    terrain.is_plane_terrain[:] = False
    state = _state(batch=1)
    state.root_pos[:, 2] = 1.02
    state.foot_pos[..., 2] = 0.70
    command = torch.tensor([[0.55, 0.0, 0.0]], dtype=torch.float32)

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 1
    cfg.runtime.terrain_height_variation_threshold_m = 1.0
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert result.loss_breakdown["qp_continuous_solver_reachability_update_count"].item() > 0
    assert result.loss_breakdown["qp_continuous_fk_readback_error_max"].item() <= 0.05


def test_mpc_qp_continuous_fk_readback_lowers_root_when_feet_float_without_repair() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    state.root_pos[:, 2] = 0.62
    state.foot_pos[..., 2] = 0.0
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 3
    cfg.runtime.continuous_plane_root_height_max_m = 0.70
    cfg.runtime.continuous_fk_root_z_max_step_m = 0.18
    baseline_cfg = MpcQpPlannerCfg()
    baseline_cfg.runtime.qp_iterations = 3
    baseline_cfg.runtime.continuous_plane_root_height_max_m = 0.70
    baseline_cfg.runtime.continuous_fk_root_z_gain = 0.0
    baseline = plan_segment_qp(terrain, state, command, cfg=baseline_cfg)
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert baseline.loss_breakdown is not None
    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_repair_main_path_active"], torch.zeros(1))
    assert result.loss_breakdown["qp_continuous_solver_fk_root_z_update_count"].item() > 0
    assert result.loss_breakdown["qp_continuous_solver_fk_root_z_delta_max"].item() > 0.05
    assert result.root_pos[0, -1, 2].item() < 0.50
    assert result.loss_breakdown["qp_continuous_fk_readback_error_max"].item() < (
        baseline.loss_breakdown["qp_continuous_fk_readback_error_max"].item() * 0.5
    )
    assert result.loss_breakdown["qp_continuous_planned_foot_terrain_penetration_count"].item() == 0


def test_mpc_qp_continuous_route_does_not_modify_current_mpc_backend() -> None:
    from extension.batch_mpc_planner.config import MpcPlannerCfg
    from extension.batch_mpc_planner.planner import plan_segment
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=9)
    state = _state(batch=1)
    command = torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32)
    mpc_result = plan_segment(terrain, state, command, cfg=MpcPlannerCfg())
    qp_result = plan_segment_qp(terrain, state, command, cfg=MpcQpPlannerCfg())

    assert mpc_result.loss_breakdown is None or "qp_continuous_enabled" not in mpc_result.loss_breakdown
    assert qp_result.loss_breakdown is not None
    assert "qp_continuous_enabled" in qp_result.loss_breakdown


def test_mpc_qp_plan_segment_reports_stage_timing_diagnostics() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    terrain = _flat_terrain(batch=2)
    state = _state(batch=2)
    command = torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    for key in (
        "qp_nominal_ms",
        "qp_solve_ms",
        "qp_repair_ms",
        "qp_diagnostics_ms",
        "qp_total_ms",
    ):
        assert key in result.loss_breakdown
        values = result.loss_breakdown[key]
        assert values.shape == (2,)
        assert torch.isfinite(values).all()
        assert torch.count_nonzero(values < 0.0) == 0
    assert torch.count_nonzero(result.loss_breakdown["qp_total_ms"] < result.loss_breakdown["qp_solve_ms"]) == 0


def test_mpc_qp_viewer_crossing_probe_summary_requires_foot_over_and_no_contact() -> None:
    from mpc_qp_viewer_crossing_probe import parse_command_list, summarize_viewer_crossing_rows

    assert parse_command_list("forward:0.45,0.0,0.0;left:0.0,0.30,0.0") == (
        "forward:0.45,0.0,0.0",
        "left:0.0,0.30,0.0",
    )

    rows = [
        {
            "crossing_leg_count": 1,
            "fk_foot_over_low_small_success": 1,
            "fk_semantic_collision_count": 0,
            "fk_foot_small_penetration_rate": 0.0,
            "fk_touchdown_on_small_rate": 0.0,
            "playback_readback_error_max_m": 0.0,
            "qp_continuous_foot_frame_jump_max": 0.08,
            "qp_continuous_joint_frame_jump_max": 0.20,
            "qp_continuous_fk_readback_error_max": 0.01,
        },
        {
            "crossing_leg_count": 1,
            "fk_foot_over_low_small_success": 1,
            "fk_semantic_collision_count": 0,
            "fk_foot_small_penetration_rate": 0.0,
            "fk_touchdown_on_small_rate": 0.0,
            "playback_readback_error_max_m": 1.0e-6,
            "qp_continuous_foot_frame_jump_max": 0.09,
            "qp_continuous_joint_frame_jump_max": 0.21,
            "qp_continuous_fk_readback_error_max": 0.02,
        },
    ]

    summary = summarize_viewer_crossing_rows(rows)

    assert summary["cycle_count"] == 2
    assert summary["fk_foot_over_low_small_success_count"] == 2
    assert summary["max_fk_semantic_collision_count"] == 0
    assert summary["max_fk_foot_small_penetration_rate"] == 0.0
    assert summary["max_fk_touchdown_on_small_rate"] == 0.0
    assert summary["max_playback_readback_error_m"] == 1.0e-6
    assert summary["max_qp_continuous_foot_frame_jump_m"] == 0.09
    assert summary["max_qp_continuous_joint_frame_jump_rad"] == 0.21
    assert summary["max_qp_continuous_fk_readback_error_m"] == 0.02
    assert summary["viewer_crossing_acceptance_passed"] is True

    bad = summarize_viewer_crossing_rows(rows + [{"crossing_leg_count": 1, "fk_foot_over_low_small_success": 0}])
    assert bad["viewer_crossing_acceptance_passed"] is False

    jumpy = summarize_viewer_crossing_rows(rows + [{"crossing_leg_count": 1, "fk_foot_over_low_small_success": 1, "qp_continuous_foot_frame_jump_max": 0.35}])
    assert jumpy["viewer_crossing_acceptance_passed"] is False

    bad_readback = summarize_viewer_crossing_rows(
        rows + [{"crossing_leg_count": 1, "fk_foot_over_low_small_success": 1, "playback_readback_error_max_m": 0.08}]
    )
    assert bad_readback["viewer_crossing_acceptance_passed"] is False

    joint_jumpy = summarize_viewer_crossing_rows(
        rows + [{"crossing_leg_count": 1, "fk_foot_over_low_small_success": 1, "qp_continuous_joint_frame_jump_max": 1.35}]
    )
    assert joint_jumpy["viewer_crossing_acceptance_passed"] is False


def test_real_viewer_runtime_fixture_can_keep_full_semantic_terrain_grid_for_hard_terrain_probe() -> None:
    from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture

    def make_env_cfg() -> SimpleNamespace:
        return SimpleNamespace(
            scene=SimpleNamespace(
                semantic_height_scanner=object(),
                terrain=SimpleNamespace(
                    max_init_terrain_level=9,
                    terrain_generator=SimpleNamespace(num_rows=10, num_cols=20),
                ),
            ),
        )

    default_fixture = object.__new__(RealViewerRuntimeFixture)
    default_fixture.env_cfg = make_env_cfg()
    default_fixture._compact_semantic_grid = True
    default_fixture._configure_compact_semantic_runtime_grid()
    default_gen = default_fixture.env_cfg.scene.terrain.terrain_generator

    full_fixture = object.__new__(RealViewerRuntimeFixture)
    full_fixture.env_cfg = make_env_cfg()
    full_fixture._compact_semantic_grid = False
    full_fixture._configure_compact_semantic_runtime_grid()
    full_gen = full_fixture.env_cfg.scene.terrain.terrain_generator

    assert (default_gen.num_rows, default_gen.num_cols) == (4, 1)
    assert (full_gen.num_rows, full_gen.num_cols) == (10, 20)


def test_real_viewer_runtime_fixture_moves_root_to_selected_tile_before_hard_terrain_scan() -> None:
    from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture

    fixture = object.__new__(RealViewerRuntimeFixture)
    calls: list[tuple[str, object]] = []

    def select_terrain_tile(*, terrain_row: int, terrain_col: int) -> torch.Tensor:
        calls.append(("select", (int(terrain_row), int(terrain_col))))
        return torch.tensor([12.5, -3.25, 0.0], dtype=torch.float32)

    def write_env0_root_xy(world_xy, *, z_clearance: float = 0.65) -> None:
        calls.append(("root", (tuple(float(value) for value in world_xy), float(z_clearance))))

    def sync_targeted_scan_pose() -> None:
        calls.append(("sync", None))

    fixture.select_terrain_tile = select_terrain_tile
    fixture._write_env0_root_xy = write_env0_root_xy
    fixture._sync_targeted_scan_pose = sync_targeted_scan_pose

    selected = fixture.move_env0_to_terrain_tile(terrain_row=7, terrain_col=11, z_clearance=0.72)

    torch.testing.assert_close(selected, torch.tensor([12.5, -3.25, 0.0], dtype=torch.float32))
    assert calls == [
        ("select", (7, 11)),
        ("root", ((12.5, -3.25), 0.72)),
        ("sync", None),
    ]


def test_real_viewer_runtime_fixture_moves_root_to_selected_tile_offset() -> None:
    from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture

    fixture = object.__new__(RealViewerRuntimeFixture)
    calls: list[tuple[str, object]] = []
    fixture.select_terrain_tile = lambda terrain_row, terrain_col: torch.tensor([12.5, -3.25, 0.0], dtype=torch.float32)
    fixture._write_env0_root_xy = lambda world_xy, z_clearance=0.65: calls.append(
        ("root", (tuple(float(value) for value in world_xy), float(z_clearance)))
    )
    fixture._sync_targeted_scan_pose = lambda: calls.append(("sync", None))

    selected = fixture.move_env0_to_terrain_tile(
        terrain_row=7,
        terrain_col=11,
        z_clearance=0.72,
        offset_xy_m=(0.4, -0.2),
    )

    torch.testing.assert_close(selected, torch.tensor([12.5, -3.25, 0.0], dtype=torch.float32))
    assert calls == [
        ("root", ((12.9, -3.45), 0.72)),
        ("sync", None),
    ]


def test_real_viewer_runtime_fixture_can_ground_root_after_selected_tile_move() -> None:
    from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture

    fixture = object.__new__(RealViewerRuntimeFixture)
    fixture.base_env = SimpleNamespace()
    fixture.scanner = object()
    fixture.foot_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    calls: list[str] = []
    fixture.select_terrain_tile = lambda terrain_row, terrain_col: torch.tensor([1.0, 2.0, -0.5])
    fixture._write_env0_root_xy = lambda world_xy, z_clearance=0.65: calls.append("root")
    fixture._sync_targeted_scan_pose = lambda: calls.append("sync")
    fixture._viewer = SimpleNamespace(
        _viewer_ground_robot_from_scanner=lambda base_env, scanner, foot_ids: calls.append(f"ground:{tuple(foot_ids.tolist())}")
        or -2.25
    )

    selected = fixture.move_env0_to_terrain_tile(terrain_row=9, terrain_col=19, ground_robot=True)

    torch.testing.assert_close(selected, torch.tensor([1.0, 2.0, -0.5]))
    assert calls == ["root", "sync", "ground:(1, 2, 3, 4)", "sync"]


def test_mpc_qp_hard_terrain_probe_uses_tile_move_contract() -> None:
    probe_path = REPO_ROOT / "Go2Pvcnn" / "tests" / "mpc_qp_hard_terrain_probe.py"
    source = probe_path.read_text(encoding="utf-8")

    assert "move_env0_to_terrain_tile" in source
    assert "ground_robot=True" in source
    assert "viewer_hard_terrain_acceptance_passed" in source
    assert "qp_continuous_root_terrain_risk_reduces_progress" in source
    assert "qp_continuous_root_height_variation_max" in source
    assert "max_qp_continuous_foot_frame_jump_m" in source


def test_mpc_qp_hard_terrain_probe_command_parser_preserves_comma_triples() -> None:
    from mpc_qp_hard_terrain_probe import _parse_command_list

    commands = _parse_command_list("forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0")

    assert commands == ("forward:0.35,0.0,0.0", "diag_left:0.30,0.12,0.0")


def test_mpc_qp_hard_terrain_probe_offset_parser_preserves_xy_pairs() -> None:
    from mpc_qp_hard_terrain_probe import _parse_offsets

    offsets = _parse_offsets("0:0,0.4:-0.2,-0.4:0.2")

    assert offsets == ((0.0, 0.0), (0.4, -0.2), (-0.4, 0.2))


def test_viewer_compute_mpc_local_terrain_uses_zero_semantics_when_scanner_has_no_semantic_map(monkeypatch) -> None:
    from extension.viz import go2_foostep_planner as viewer

    ray_hits = torch.zeros((1, 9, 3), dtype=torch.float32)
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=ray_hits,
            pos_w=torch.zeros((1, 3), dtype=torch.float32),
            quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
        cfg=SimpleNamespace(pattern_cfg=SimpleNamespace(size=(1.0, 1.0))),
    )
    monkeypatch.setattr(viewer, "_scanner_semantic_map", lambda scanner, env_id=0: None)

    terrain, _hits = viewer._compute_mpc_local_terrain(scanner, env_id=0)

    assert terrain.semantic_map.shape == terrain.height_map.shape
    assert torch.count_nonzero(terrain.semantic_map).item() == 0


def test_mpc_qp_viewer_crossing_summary_ignores_rows_without_crossing_opportunity() -> None:
    from mpc_qp_viewer_crossing_probe import summarize_viewer_crossing_rows

    rows = [
        {
            "crossing_leg_count": 1,
            "fk_foot_over_low_small_success": 1,
            "fk_semantic_collision_count": 0,
            "fk_foot_small_penetration_rate": 0.0,
            "fk_touchdown_on_small_rate": 0.0,
        },
        {
            "crossing_leg_count": 0,
            "fk_foot_over_low_small_success": 0,
            "fk_semantic_collision_count": 0,
            "fk_foot_small_penetration_rate": 0.0,
            "fk_touchdown_on_small_rate": 0.0,
        },
    ]

    summary = summarize_viewer_crossing_rows(rows)

    assert summary["crossing_opportunity_count"] == 1
    assert summary["fk_foot_over_low_small_required_success_count"] == 1
    assert summary["viewer_crossing_acceptance_passed"] is True

    unsafe = summarize_viewer_crossing_rows(rows + [{"crossing_leg_count": 0, "fk_semantic_collision_count": 1}])
    assert unsafe["viewer_crossing_acceptance_passed"] is False


def test_mpc_qp_executes_configured_iteration_count() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 3
    terrain = _flat_terrain(batch=2)
    state = _state(batch=2)
    command = torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float32)

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert torch.equal(result.loss_breakdown["qp_iterations_configured"], torch.full((2,), 3.0))
    assert torch.equal(result.loss_breakdown["qp_iterations_executed"], torch.full((2,), 3.0))


def test_mpc_qp_semantic_qp_update_prefers_nearby_safe_touchdown_over_anchor_fallback() -> None:
    from extension.batch_mpc_planner.terrain import semantic_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.25, 0.0, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    nominal_leg0_xy = baseline.planned_touchdown_w[0, 0, 0, :2].detach()
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - nominal_leg0_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - nominal_leg0_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1

    repaired = plan_segment_qp(terrain, state, command, cfg=cfg)
    repaired_leg0 = repaired.planned_touchdown_w[0, 0, 0]
    anchor_leg0 = state.foot_pos[0, 0]
    touchdown_semantic = semantic_at(terrain, repaired.planned_touchdown_w[:, 0, :, :2])

    assert int(touchdown_semantic[0, 0].item()) == 0
    assert torch.linalg.vector_norm(repaired_leg0[:2] - anchor_leg0[:2]) > 0.03
    assert repaired.loss_breakdown is not None
    assert repaired.loss_breakdown["qp_touchdown_on_small_count"].item() == 0
    assert repaired.loss_breakdown["qp_max_semantic_constraint_violation"].item() <= 1.0e-5


def test_mpc_qp_high_height_variation_reduces_root_progress_and_reports_step_cap() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    state = _state(batch=1)
    command = torch.tensor([[0.40, 0.0, 0.0]], dtype=torch.float32)
    flat = _flat_terrain(batch=1, size=41)
    rough = _flat_terrain(batch=1, size=41)
    rough.height_map[:, :, 22:] = 0.22

    flat_result = plan_segment_qp(flat, state, command, cfg=cfg)
    rough_result = plan_segment_qp(rough, state, command, cfg=cfg)
    flat_progress = flat_result.root_pos[0, -1, 0] - state.root_pos[0, 0]
    rough_progress = rough_result.root_pos[0, -1, 0] - state.root_pos[0, 0]

    assert rough_progress < flat_progress * 0.85
    assert rough_result.loss_breakdown is not None
    assert rough_result.loss_breakdown["qp_terrain_risk_reduces_target_progress"].item() == 1
    assert rough_result.loss_breakdown["qp_step_cap_violation_count"].item() == 0


def test_mpc_qp_reports_low_small_crossing_without_touchdown_or_path_collision() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    leg0_path = baseline.foot_pos[0, :, 0, :2].detach()
    mid_xy = leg0_path[len(leg0_path) // 2]
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - mid_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - mid_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 0.05

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_crossing_leg_count"].item() > 0
    assert result.loss_breakdown["qp_touchdown_on_small_count"].item() == 0
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_semantic_collision_rate"].item() == 0
    assert result.loss_breakdown["qp_fk_semantic_min_clearance_over_semantic_m"].item() >= 0


def test_mpc_qp_lateral_command_low_small_crossing_keeps_fk_leg_over_obstacle() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.0, 0.35, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    leg1_path = baseline.foot_pos[0, :, 1, :2].detach()
    mid_xy = leg1_path[len(leg1_path) // 2]
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - mid_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - mid_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 0.05

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_crossing_leg_count"].item() > 0
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_body_leg_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_semantic_min_clearance_over_semantic_m"].item() >= 0
    assert result.loss_breakdown["qp_low_small_swing_over_repair_count"].item() > 0


def test_mpc_qp_diag_fl_low_small_crossing_lifts_then_lands() -> None:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp
    from mpc_low_small_reachable_crossing_probe import reachable_foot_over_arc_metrics

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.32, 0.32, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    leg0_path = baseline.foot_pos[0, :, 0, :2].detach()
    mid_xy = leg0_path[len(leg0_path) // 2]
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - mid_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - mid_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 0.05
    obstacle_xy = torch.stack((xs[ix], ys[iy]))

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    metrics = reachable_foot_over_arc_metrics(
        fk_foot,
        result.contact_state,
        obstacle_xy,
        command=(0.32, 0.32, 0.0),
        obstacle_height=0.05,
        clearance=0.05,
        lane_half_width=0.10,
    )

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_touchdown_on_small_count"].item() == 0
    assert metrics["fk_foot_over_low_small_lift_then_land"] == 1
    assert metrics["fk_foot_over_low_small_touchdown_after"] == 1
    assert metrics["fk_foot_over_low_small_success"] == 1


def test_mpc_qp_diag_fl_contact_phase_low_small_crossing_relands_after_obstacle() -> None:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp
    from mpc_low_small_reachable_crossing_probe import reachable_foot_over_arc_metrics

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.32, 0.32, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    contact_frames = torch.nonzero(baseline.contact_state[0, :, 0], as_tuple=False).reshape(-1)
    frame_idx = int(contact_frames[-1].item())
    obstacle_xy = baseline.foot_pos[0, frame_idx, 0, :2].detach()
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - obstacle_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - obstacle_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 0.05
    obstacle_center = torch.stack((xs[ix], ys[iy]))

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    metrics = reachable_foot_over_arc_metrics(
        fk_foot,
        result.contact_state,
        obstacle_center,
        command=(0.32, 0.32, 0.0),
        obstacle_height=0.05,
        clearance=0.05,
        lane_half_width=0.10,
    )

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_low_small_contact_over_repair_count"].item() > 0
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_touchdown_on_small_count"].item() == 0
    assert metrics["fk_foot_over_low_small_lift_then_land"] == 1
    assert metrics["fk_foot_over_low_small_touchdown_after"] == 1
    assert metrics["fk_foot_over_low_small_success"] == 1


def test_mpc_qp_low_small_swing_repair_updates_reland_contact_schedule() -> None:
    from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles
    from extension.batch_mpc_planner.parametric import command_frame_axes
    from extension.batch_mpc_planner.terrain import semantic_at
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.32, 0.32, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    leg0_path = baseline.foot_pos[0, :, 0, :2].detach()
    mid_xy = leg0_path[len(leg0_path) // 2]
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - mid_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - mid_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 0.05

    result = plan_segment_qp(terrain, state, command, cfg=cfg)
    fk_foot = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)
    heading, _, _ = command_frame_axes(command, result.root_rpy[:, 0, 2], linear_eps=1.0e-6)
    obstacle_center = torch.stack((xs[ix], ys[iy])).to(dtype=fk_foot.dtype)
    along = ((fk_foot[..., :2] - obstacle_center.view(1, 1, 1, 2)) * heading[:, None, None, :]).sum(dim=-1)
    lane = torch.abs(
        ((fk_foot[..., :2] - obstacle_center.view(1, 1, 1, 2))
         * torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)[:, None, None, :]).sum(dim=-1)
    ) <= 0.10
    foot_semantic = semantic_at(terrain, fk_foot[..., :2])

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_low_small_swing_over_repair_count"].item() > 0
    assert torch.count_nonzero(torch.logical_and(result.contact_state, foot_semantic == 1)) == 0
    assert torch.count_nonzero(torch.logical_and(result.contact_state, torch.logical_and(along > 0.02, lane))) > 0


def test_mpc_qp_lifts_contact_leg_when_low_small_lies_on_crossing_path() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.32, 0.32, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    contact = baseline.contact_state[0]
    contact_counts = contact.to(dtype=torch.long).sum(dim=0)
    leg_idx = int(torch.argmax(contact_counts).item())
    frame_idx = int(torch.nonzero(contact[:, leg_idx], as_tuple=False).reshape(-1)[0].item())
    obstacle_xy = baseline.foot_pos[0, frame_idx, leg_idx, :2].detach()
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - obstacle_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - obstacle_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 0.05

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_low_small_contact_over_repair_count"].item() > 0
    assert result.loss_breakdown["qp_fk_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_semantic_min_clearance_over_semantic_m"].item() >= 0


def test_mpc_qp_mixed_turn_repairs_knee_semantic_collision_with_xy_avoidance() -> None:
    from extension.batch_mpc_planner.kinematics import fk_leg_points_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.45, 0.15, 0.60]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    baseline_fk = fk_leg_points_from_joint_angles(
        baseline.root_pos,
        baseline.root_rpy,
        baseline.joint_angles,
        shank_sample_count=2,
    )
    knee = baseline_fk.knee_pos_world[0]
    flat_idx = int(torch.argmin(knee[..., 2].reshape(-1)).item())
    frame_idx = flat_idx // 4
    leg_idx = flat_idx % 4
    obstacle_xy = knee[frame_idx, leg_idx, :2].detach()
    obstacle_height = float(knee[frame_idx, leg_idx, 2].item() + 0.04)
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - obstacle_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - obstacle_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = obstacle_height

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_fk_knee_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_body_leg_collision_count"].item() == 0
    assert "qp_fk_body_leg_xy_repair_count" in result.loss_breakdown
    assert "qp_fk_body_leg_root_lift_count" in result.loss_breakdown


def test_mpc_qp_continuous_solver_reduces_fk_body_leg_semantic_collision_without_repair() -> None:
    from extension.batch_mpc_planner.kinematics import fk_leg_points_from_joint_angles
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.45, 0.0, 0.0]], dtype=torch.float32)
    baseline_cfg = MpcQpPlannerCfg()
    baseline_cfg.runtime.qp_iterations = 1
    baseline = plan_segment_qp(terrain, state, command, cfg=baseline_cfg)
    baseline_fk = fk_leg_points_from_joint_angles(
        baseline.root_pos,
        baseline.root_rpy,
        baseline.joint_angles,
        shank_sample_count=2,
    )
    shank = baseline_fk.shank_sample_world[0]
    flat_idx = int(torch.argmin(shank[..., 2].reshape(-1)).item())
    frame_leg_sample = shank.reshape(-1, 3)[flat_idx]
    obstacle_xy = frame_leg_sample[:2].detach()
    obstacle_height = float(frame_leg_sample[2].item() + 0.04)
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - obstacle_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - obstacle_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = obstacle_height

    cfg = MpcQpPlannerCfg()
    cfg.runtime.qp_iterations = 2
    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_repair_main_path_active"].item() == 0
    assert "qp_continuous_solver_body_leg_clearance_update_count" in result.loss_breakdown
    assert result.loss_breakdown["qp_fk_body_leg_collision_count"].item() == 0


def test_mpc_qp_reports_fk_leg_and_underbody_collision_free_metrics() -> None:
    from extension.batch_mpc_qp_planner.config import MpcQpPlannerCfg
    from extension.batch_mpc_qp_planner.planner import plan_segment_qp

    cfg = MpcQpPlannerCfg()
    cfg.runtime.continuous_trajectory_enabled = False
    terrain = _flat_terrain(batch=1, size=41)
    state = _state(batch=1)
    command = torch.tensor([[0.30, 0.0, 0.0]], dtype=torch.float32)
    baseline = plan_segment_qp(terrain, state, command, cfg=cfg)
    leg0_path = baseline.foot_pos[0, :, 0, :2].detach()
    mid_xy = leg0_path[len(leg0_path) // 2]
    xs = torch.linspace(terrain.world_x_range[0], terrain.world_x_range[1], terrain.semantic_map.shape[-1])
    ys = torch.linspace(terrain.world_y_range[0], terrain.world_y_range[1], terrain.semantic_map.shape[-2])
    ix = int(torch.argmin(torch.abs(xs - mid_xy[0])).item())
    iy = int(torch.argmin(torch.abs(ys - mid_xy[1])).item())
    terrain.semantic_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 1
    terrain.height_map[:, max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = 0.05

    result = plan_segment_qp(terrain, state, command, cfg=cfg)

    assert result.loss_breakdown is not None
    assert result.loss_breakdown["qp_fk_body_leg_collision_count"].item() == 0
    assert result.loss_breakdown["qp_root_underbody_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_knee_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_shank_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_underbody_semantic_collision_count"].item() == 0
    assert result.loss_breakdown["qp_fk_body_leg_height_violation_max"].item() <= 1.0e-5
