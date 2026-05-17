from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.convention import euler_to_quat_batch
from extension.batched_together_planner.parameterization import (
    T116_MODE_APPROACH_SMALL,
    T116_MODE_BYPASS_OBSTACLE,
    T116_MODE_CROSS_SMALL,
    T116_MODE_CRUISE,
)
from extension.batched_together_planner.planner import (
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    DIRECTION_LATERAL_LEFT,
    DIRECTION_LATERAL_RIGHT,
)
from extension.batched_together_planner.types import TogetherPlannerStatus
from extension.batched_planner.types import LEG_ORDER
from tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from tests.fixtures.viewer_runtime_diagnostics import build_command_cases, scanner_sync_steps


def _make_real_runtime_fixture(**kwargs):
    assert hasattr(viewer_diag, "make_real_runtime_fixture")
    return viewer_diag.make_real_runtime_fixture(**kwargs)


def _t116_grounded_diag(
    *,
    mode: int,
    status: int = int(TogetherPlannerStatus.OK),
    feasible: bool = True,
    safe_fallback: bool = False,
    selected_beta: float = 1.0,
    selected_route: int = 0,
    direction_id: int = DIRECTION_FORWARD,
    command_direction_violation: bool = False,
    cross_small_success: bool = False,
    body_min_clearance: float = 0.05,
    leg_min_clearance: float = 0.05,
    base_min_clearance_to_small: float = 0.05,
    touchdown_ground_gap_by_leg: tuple[float, float, float, float] = (0.01, -0.02, 0.01, -0.01),
    per_leg_touchdown_on_small_count: tuple[int, int, int, int] = (0, 0, 0, 0),
    per_leg_foot_small_collision_count: tuple[int, int, int, int] = (0, 0, 0, 0),
    per_leg_min_clearance_to_small: tuple[float, float, float, float] = (0.03, 0.03, 0.04, 0.04),
    per_leg_touchdown_beyond_small_back_edge: tuple[bool, bool, bool, bool] = (True, True, True, True),
    touchdown_semantic_by_leg: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> viewer_diag.GroundedCrossingDiagnostics:
    gap = torch.tensor([touchdown_ground_gap_by_leg], dtype=torch.float64)
    touchdown_small = torch.tensor([per_leg_touchdown_on_small_count], dtype=torch.int64)
    foot_collision = torch.tensor([per_leg_foot_small_collision_count], dtype=torch.int64)
    return viewer_diag.GroundedCrossingDiagnostics(
        mode=torch.tensor([mode], dtype=torch.int64),
        status=torch.tensor([status], dtype=torch.int64),
        feasible=torch.tensor([feasible], dtype=torch.bool),
        safe_fallback=torch.tensor([safe_fallback], dtype=torch.bool),
        selected_beta=torch.tensor([selected_beta], dtype=torch.float64),
        selected_route=torch.tensor([selected_route], dtype=torch.int64),
        direction_id=torch.tensor([direction_id], dtype=torch.int64),
        command_direction_violation=torch.tensor([command_direction_violation], dtype=torch.bool),
        cross_small_success=torch.tensor([cross_small_success], dtype=torch.bool),
        body_min_clearance=torch.tensor([body_min_clearance], dtype=torch.float64),
        leg_min_clearance=torch.tensor([leg_min_clearance], dtype=torch.float64),
        base_min_clearance_to_small=torch.tensor([base_min_clearance_to_small], dtype=torch.float64),
        per_leg_touchdown_on_small_count=touchdown_small,
        per_leg_foot_small_collision_count=foot_collision,
        per_leg_min_clearance_to_small=torch.tensor([per_leg_min_clearance_to_small], dtype=torch.float64),
        per_leg_touchdown_beyond_small_back_edge=torch.tensor(
            [per_leg_touchdown_beyond_small_back_edge],
            dtype=torch.bool,
        ),
        touchdown_ground_gap_by_leg=gap,
        touchdown_semantic_by_leg=torch.tensor([touchdown_semantic_by_leg], dtype=torch.int64),
        state_mode=torch.tensor([mode], dtype=torch.int64),
        small_strategy_outcome=torch.tensor([mode], dtype=torch.int64),
        front_touchdown_ground_gap=gap[:, :2].clone(),
        rear_touchdown_ground_gap=gap[:, 2:].clone(),
        touchdown_on_small_count=touchdown_small.sum(dim=-1),
        front_foot_small_collision_count=foot_collision[:, :2].sum(dim=-1),
        rear_foot_small_collision_count=foot_collision[:, 2:].sum(dim=-1),
        base_small_penetration_count=torch.tensor([int(base_min_clearance_to_small < 0.0)], dtype=torch.int64),
        base_path_crosses_small_flag=torch.tensor([base_min_clearance_to_small < 0.0], dtype=torch.bool),
    )


@pytest.fixture(scope="module")
def real_runtime():
    runtime = _make_real_runtime_fixture(num_envs=2)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture(scope="module")
def real_batched_runtime(real_runtime):
    return real_runtime


@pytest.fixture(scope="module")
def real_semantic_together_runtime():
    runtime = _make_real_runtime_fixture(num_envs=2, planner_backend="together")
    try:
        yield runtime
    finally:
        runtime.close()


def test_build_command_cases_includes_forward_command():
    cases = build_command_cases(device=torch.device("cpu"), num_envs=1)

    assert "forward" in cases
    assert cases["forward"].shape == (1, 3)
    assert torch.linalg.vector_norm(cases["forward"]).item() > 0


def test_scanner_sync_steps_waits_past_update_period():
    assert scanner_sync_steps(scanner_update_period=0.02, physics_dt=0.005, minimum_steps=1) == 8
    assert scanner_sync_steps(scanner_update_period=0.0, physics_dt=0.005, minimum_steps=3) == 4


def test_refresh_targeted_scanner_pose_uses_viewer_refresh_helper():
    calls = {"render": 0, "update": 0}

    class FakeSim:
        def render(self):
            calls["render"] += 1

    class FakeScene:
        def update(self, _dt):
            calls["update"] += 1

    base_env = type("FakeBaseEnv", (), {"sim": FakeSim(), "scene": FakeScene(), "physics_dt": 0.005})()
    scanner = type("FakeScanner", (), {"cfg": type("Cfg", (), {"update_period": 0.02})()})()

    steps = viewer_diag.refresh_targeted_scanner_pose(base_env, scanner, minimum_steps=1)

    assert steps == 8
    assert calls["render"] == 8
    assert calls["update"] == 8


def test_runtime_resource_error_detection_requires_resource_evidence():
    assert viewer_diag._is_runtime_resource_error(AttributeError("'Articulation' object has no attribute '_data'")) is False
    assert (
        viewer_diag._is_runtime_resource_error(
            RuntimeError(
                "Unable to allocate memory of size 671088640 for mGpuContactPairsDev; "
                "'Articulation' object has no attribute '_data'"
            )
        )
        is True
    )


def test_runtime_app_launcher_init_failure_closes_partial_app_and_clears_state(monkeypatch):
    closed = {"value": False}

    class FakeApp:
        def close(self):
            closed["value"] = True

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true", default=False)
            parser.add_argument("--device", type=str, default="cuda:0")

        def __init__(self, args_cli):
            self.app = FakeApp()
            raise RuntimeError("launcher init failed")

    fake_isaaclab = ModuleType("isaaclab")
    fake_app_module = ModuleType("isaaclab.app")
    fake_app_module.AppLauncher = FakeAppLauncher
    fake_isaaclab.app = fake_app_module

    monkeypatch.setattr(viewer_diag, "_APP_STATE", None)
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app_module)

    with pytest.raises(RuntimeError, match="launcher init failed"):
        viewer_diag._ensure_runtime_app(device="cuda:0")

    assert closed["value"] is True
    assert viewer_diag._APP_STATE is None


def test_real_runtime_fixture_close_leaves_shared_app_alive(monkeypatch):
    closed = {"env": False, "app": False}

    class FakeEnv:
        def close(self):
            closed["env"] = True

    class FakeApp:
        def close(self):
            closed["app"] = True

    runtime = viewer_diag.RealViewerRuntimeFixture.__new__(viewer_diag.RealViewerRuntimeFixture)
    runtime._closed = False
    runtime.env = FakeEnv()
    monkeypatch.setattr(viewer_diag, "_APP_STATE", viewer_diag._RuntimeAppState(launcher=object(), app=FakeApp(), device="cuda:0"))

    runtime.close()

    assert closed["env"] is True
    assert closed["app"] is False
    assert runtime._closed is True
    assert viewer_diag._APP_STATE is not None


def test_together_viewer_adapter_preserves_grounded_crossing_fields():
    from extension.viz import go2_foostep_planner as viewer_module

    root_rpy = torch.zeros((1, 3, 3), dtype=torch.float64)
    expected_quat = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2])
    result = type(
        "FakeTogetherResult",
        (),
        {
            "root_pos": torch.zeros((1, 3, 3), dtype=torch.float64),
            "root_rpy": root_rpy,
            "foot_pos": torch.zeros((1, 3, 4, 3), dtype=torch.float64),
            "touchdown_seq": torch.zeros((1, 4, 2, 3), dtype=torch.float64),
            "joint_angles": torch.zeros((1, 3, 12), dtype=torch.float64),
            "contact_state": torch.ones((1, 3, 4), dtype=torch.bool),
            "mode": torch.tensor([T116_MODE_CROSS_SMALL], dtype=torch.int64),
            "status": torch.tensor([int(TogetherPlannerStatus.OK)], dtype=torch.int64),
            "feasible": torch.tensor([True], dtype=torch.bool),
            "safe_fallback": torch.tensor([False], dtype=torch.bool),
            "selected_beta": torch.tensor([0.75], dtype=torch.float64),
            "selected_route": torch.tensor([0], dtype=torch.int64),
            "direction_id": torch.tensor([0], dtype=torch.int64),
            "command_direction_violation": torch.tensor([False], dtype=torch.bool),
            "cross_small_success": torch.tensor([True], dtype=torch.bool),
            "body_min_clearance": torch.tensor([0.06], dtype=torch.float64),
            "leg_min_clearance": torch.tensor([0.04], dtype=torch.float64),
            "base_min_clearance_to_small": torch.tensor([0.05], dtype=torch.float64),
            "per_leg_touchdown_on_small_count": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64),
            "per_leg_foot_small_collision_count": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64),
            "per_leg_min_clearance_to_small": torch.tensor([[0.03, 0.04, 0.05, 0.06]], dtype=torch.float64),
            "per_leg_touchdown_beyond_small_back_edge": torch.tensor([[True, True, True, True]], dtype=torch.bool),
            "touchdown_ground_gap_by_leg": torch.tensor([[0.01, -0.02, 0.03, -0.04]], dtype=torch.float64),
            "touchdown_semantic_by_leg": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64),
        },
    )()

    adapted = viewer_module._adapt_together_result_for_viewer(result)

    torch.testing.assert_close(adapted.root_quat_w, expected_quat, atol=0.0, rtol=0.0)
    assert adapted.mode is not None
    assert adapted.status is not None
    assert adapted.feasible is not None
    assert adapted.safe_fallback is not None
    assert adapted.selected_beta is not None
    assert adapted.selected_route is not None
    assert adapted.direction_id is not None
    assert adapted.cross_small_success is not None
    assert adapted.body_min_clearance is not None
    assert adapted.leg_min_clearance is not None
    assert adapted.base_min_clearance_to_small is not None
    assert adapted.touchdown_ground_gap_by_leg is not None
    assert adapted.touchdown_semantic_by_leg is not None
    assert int(adapted.mode.item()) == T116_MODE_CROSS_SMALL
    assert int(adapted.status.item()) == int(TogetherPlannerStatus.OK)
    assert bool(adapted.feasible.item()) is True
    assert bool(adapted.cross_small_success.item()) is True
    assert int(adapted.direction_id.item()) == DIRECTION_FORWARD
    torch.testing.assert_close(adapted.selected_beta, torch.tensor([0.75], dtype=torch.float64))
    torch.testing.assert_close(adapted.touchdown_ground_gap_by_leg, result.touchdown_ground_gap_by_leg)


def test_runtime_plan_diagnostics_builds_grounded_crossing_wrapper():
    fixture = viewer_diag.RealViewerRuntimeFixture.__new__(viewer_diag.RealViewerRuntimeFixture)
    fixture._viewer = type(
        "FakeViewer",
        (),
        {
            "_trajectory_motion_summary": staticmethod(lambda _result: {"dx": 0.1, "dy": 0.0, "dz": 0.0, "dyaw": 0.0, "standstill": False}),
        },
    )()
    fixture._single_env_terrain_and_hits = lambda: (None, torch.zeros((1, 2, 2, 3), dtype=torch.float64))
    fixture._semantic_scan_diagnostics = lambda _ray_hits: {
        "valid_sample_count": 1,
        "terrain_hit_count": 1,
        "small_hit_count": 0,
        "large_hit_count": 0,
        "height_lift_max": 0.0,
    }
    state = type("FakeState", (), {"foot_pos": torch.zeros((1, 4, 3), dtype=torch.float64)})()
    result = type(
        "FakeViewerResult",
        (),
        {
            "planned_touchdown_w": torch.zeros((1, 4, 3), dtype=torch.float64),
            "mode": torch.tensor([T116_MODE_CROSS_SMALL], dtype=torch.int64),
            "status": torch.tensor([int(TogetherPlannerStatus.OK)], dtype=torch.int64),
            "feasible": torch.tensor([True], dtype=torch.bool),
            "safe_fallback": torch.tensor([False], dtype=torch.bool),
            "selected_beta": torch.tensor([0.5], dtype=torch.float64),
            "selected_route": torch.tensor([1], dtype=torch.int64),
            "direction_id": torch.tensor([DIRECTION_FORWARD], dtype=torch.int64),
            "command_direction_violation": torch.tensor([False], dtype=torch.bool),
            "cross_small_success": torch.tensor([True], dtype=torch.bool),
            "body_min_clearance": torch.tensor([0.07], dtype=torch.float64),
            "leg_min_clearance": torch.tensor([0.05], dtype=torch.float64),
            "base_min_clearance_to_small": torch.tensor([0.06], dtype=torch.float64),
            "per_leg_touchdown_on_small_count": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64),
            "per_leg_foot_small_collision_count": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64),
            "per_leg_min_clearance_to_small": torch.tensor([[0.03, 0.04, 0.05, 0.06]], dtype=torch.float64),
            "per_leg_touchdown_beyond_small_back_edge": torch.tensor([[True, True, True, True]], dtype=torch.bool),
            "touchdown_ground_gap_by_leg": torch.tensor([[0.01, 0.02, 0.03, 0.04]], dtype=torch.float64),
            "touchdown_semantic_by_leg": torch.tensor([[0, 0, 0, 0]], dtype=torch.int64),
        },
    )()

    diagnostics = fixture._build_runtime_plan_diagnostics(
        name="forward",
        command=torch.zeros((1, 3), dtype=torch.float64),
        state=state,
        result=result,
    )

    assert diagnostics.grounded_crossing is not None
    assert diagnostics.grounded_crossing_summary == {
        "mode": T116_MODE_CROSS_SMALL,
        "status": int(TogetherPlannerStatus.OK),
        "feasible": True,
        "safe_fallback": False,
        "selected_beta": 0.5,
        "selected_route": 1,
        "direction_id": DIRECTION_FORWARD,
        "state_mode": T116_MODE_CROSS_SMALL,
        "small_strategy_outcome": T116_MODE_CROSS_SMALL,
        "command_direction_violation": False,
        "cross_small_success": True,
        "body_min_clearance": 0.07,
        "leg_min_clearance": 0.05,
        "base_min_clearance_to_small": 0.06,
        "per_leg_touchdown_on_small_count": (0, 0, 0, 0),
        "per_leg_foot_small_collision_count": (0, 0, 0, 0),
        "per_leg_min_clearance_to_small": (0.03, 0.04, 0.05, 0.06),
        "per_leg_touchdown_beyond_small_back_edge": (True, True, True, True),
        "touchdown_ground_gap_by_leg": (0.01, 0.02, 0.03, 0.04),
        "touchdown_semantic_by_leg": (0, 0, 0, 0),
        "front_touchdown_ground_gap": (0.01, 0.02),
        "rear_touchdown_ground_gap": (0.03, 0.04),
        "touchdown_on_small_count": 0,
        "front_foot_small_collision_count": 0,
        "rear_foot_small_collision_count": 0,
        "base_small_penetration_count": 0,
        "base_path_crosses_small_flag": False,
    }


def test_cobblestone_runtime_grid_defaults_compact_and_accepts_metric_shape():
    class FakeTerrainGen:
        num_rows = 10
        num_cols = 20
        curriculum = True

    class FakeTerrainCfg:
        terrain_generator = FakeTerrainGen()
        max_init_terrain_level = 9

    class FakeScene:
        terrain = FakeTerrainCfg()

    class FakeEnvCfg:
        scene = FakeScene()

    compact = viewer_diag.RealViewerRuntimeFixture.__new__(viewer_diag.RealViewerRuntimeFixture)
    compact.terrain = "cobblestone"
    compact.env_cfg = FakeEnvCfg()
    compact._cobblestone_num_rows = None
    compact._cobblestone_num_cols = None
    compact._configure_compact_cobblestone_runtime_grid()

    assert compact.env_cfg.scene.terrain.terrain_generator.num_rows == 2
    assert compact.env_cfg.scene.terrain.terrain_generator.num_cols == 1
    assert compact.env_cfg.scene.terrain.max_init_terrain_level == 1

    metric = viewer_diag.RealViewerRuntimeFixture.__new__(viewer_diag.RealViewerRuntimeFixture)
    metric.terrain = "cobblestone"
    metric.env_cfg = FakeEnvCfg()
    metric._cobblestone_num_rows = 4
    metric._cobblestone_num_cols = 7
    metric._configure_compact_cobblestone_runtime_grid()

    assert metric.env_cfg.scene.terrain.terrain_generator.num_rows == 4
    assert metric.env_cfg.scene.terrain.terrain_generator.num_cols == 7
    assert metric.env_cfg.scene.terrain.max_init_terrain_level == 3


def test_real_runtime_fixture_selects_requested_terrain_tile():
    calls = []

    class FakeViewer:
        @staticmethod
        def _apply_viewer_terrain_selection(scene, *, env_id: int, terrain_row: int, terrain_col: int):
            calls.append((scene, env_id, terrain_row, terrain_col))
            return torch.tensor((float(terrain_row), float(terrain_col), 0.0), dtype=torch.float32)

    fake_scene = object()
    fixture = viewer_diag.RealViewerRuntimeFixture.__new__(viewer_diag.RealViewerRuntimeFixture)
    fixture._viewer = FakeViewer()
    fixture.base_env = type("FakeBaseEnv", (), {"scene": fake_scene})()
    fixture.terrain_row = 1
    fixture.terrain_col = 2

    origin = fixture.select_terrain_tile(terrain_row=3, terrain_col=4)

    assert fixture.terrain_row == 3
    assert fixture.terrain_col == 4
    torch.testing.assert_close(origin, torch.tensor((3.0, 4.0, 0.0), dtype=torch.float32))
    assert calls == [(fake_scene, 0, 3, 4)]


def test_grounded_crossing_runtime_sequence_report_summarizes_acceptance_fields():
    fixture = viewer_diag.RealViewerRuntimeFixture.__new__(viewer_diag.RealViewerRuntimeFixture)
    plans = [
        viewer_diag.RuntimePlanDiagnostics(
            name="forward",
            command=torch.zeros((1, 3), dtype=torch.float64),
            result=object(),
            summary={},
            semantic_diagnostics={},
            grounded_crossing=_t116_grounded_diag(
                mode=T116_MODE_APPROACH_SMALL,
                selected_beta=1.0,
                selected_route=0,
                cross_small_success=False,
            ),
            grounded_crossing_summary=None,
            touchdown_xy_deltas=torch.zeros((1, 4, 2), dtype=torch.float64),
            touchdown_xy_delta_norms=torch.zeros((4,), dtype=torch.float64),
            left_touchdown_mean_y=0.0,
            right_touchdown_mean_y=0.0,
        ),
        viewer_diag.RuntimePlanDiagnostics(
            name="forward",
            command=torch.zeros((1, 3), dtype=torch.float64),
            result=object(),
            summary={},
            semantic_diagnostics={},
            grounded_crossing=_t116_grounded_diag(
                mode=T116_MODE_CROSS_SMALL,
                selected_beta=0.75,
                selected_route=0,
                cross_small_success=True,
            ),
            grounded_crossing_summary=None,
            touchdown_xy_deltas=torch.zeros((1, 4, 2), dtype=torch.float64),
            touchdown_xy_delta_norms=torch.zeros((4,), dtype=torch.float64),
            left_touchdown_mean_y=0.0,
            right_touchdown_mean_y=0.0,
        ),
        viewer_diag.RuntimePlanDiagnostics(
            name="forward",
            command=torch.zeros((1, 3), dtype=torch.float64),
            result=object(),
            summary={},
            semantic_diagnostics={},
            grounded_crossing=_t116_grounded_diag(
                mode=T116_MODE_CRUISE,
                selected_beta=1.0,
                selected_route=0,
                cross_small_success=False,
            ),
            grounded_crossing_summary=None,
            touchdown_xy_deltas=torch.zeros((1, 4, 2), dtype=torch.float64),
            touchdown_xy_delta_norms=torch.zeros((4,), dtype=torch.float64),
            left_touchdown_mean_y=0.0,
            right_touchdown_mean_y=0.0,
        ),
    ]

    fixture.plan_case_near_s4_anchor_command_relative = lambda *args, **kwargs: plans.pop(0)  # type: ignore[attr-defined]
    report = fixture.grounded_crossing_runtime_sequence()

    assert report.mode_sequence == (T116_MODE_APPROACH_SMALL, T116_MODE_CROSS_SMALL, T116_MODE_CRUISE)
    assert report.state_sequence == report.mode_sequence
    assert report.small_strategy_sequence == report.mode_sequence
    assert report.status_sequence == (int(TogetherPlannerStatus.OK), int(TogetherPlannerStatus.OK), int(TogetherPlannerStatus.OK))
    assert report.feasible_sequence == (True, True, True)
    assert report.safe_fallback_sequence == (False, False, False)
    assert report.selected_beta_sequence == (1.0, 0.75, 1.0)
    assert report.selected_route_sequence == (0, 0, 0)
    assert report.direction_id_sequence == (DIRECTION_FORWARD, DIRECTION_FORWARD, DIRECTION_FORWARD)
    assert report.front_touchdown_ground_gap_abs_m == pytest.approx(0.02)
    assert report.rear_touchdown_ground_gap_abs_m == pytest.approx(0.01)
    assert report.touchdown_ground_gap_by_leg_abs_m == pytest.approx(0.02)
    assert report.rear_touchdown_airborne_count == 0
    assert report.touchdown_on_small_count == 0
    assert report.foot_small_collision_count == 0
    assert report.body_min_clearance_m == pytest.approx(0.05)
    assert report.leg_min_clearance_m == pytest.approx(0.05)
    assert report.base_min_clearance_to_small_m == pytest.approx(0.05)
    assert report.per_leg_touchdown_on_small_count == (0, 0, 0, 0)
    assert report.per_leg_foot_small_collision_count == (0, 0, 0, 0)
    assert report.per_leg_touchdown_beyond_small_back_edge == (True, True, True, True)
    assert report.touchdown_semantic_by_leg == (0, 0, 0, 0)
    assert report.command_direction_violation_count == 0
    assert report.cross_small_success_count == 1
    assert report.cross_phase_progression_valid == 1
    assert report.cross_outcome_grounded == 1


def test_viewer_forward_command_changes_plan_motion_metrics(real_runtime):
    standstill = real_runtime.plan_case("standstill")
    forward = real_runtime.plan_case("forward")

    assert standstill.summary["standstill"] is True
    assert forward.summary["standstill"] is False
    assert forward.summary["dx"] > 0.05
    assert abs(forward.summary["dx"]) > abs(forward.summary["dy"]) + 0.03


def test_viewer_runtime_uses_semantic_height_scanner_contract(real_runtime):
    assert real_runtime.scanner_name == "semantic_height_scanner"
    assert hasattr(real_runtime.scanner.data, "semantic_map")


def test_viewer_lateral_command_changes_plan_motion_metrics(real_runtime):
    standstill = real_runtime.plan_case("standstill")
    lateral = real_runtime.plan_case("lateral_left")

    assert standstill.summary["standstill"] is True
    assert lateral.summary["standstill"] is False
    assert lateral.summary["dy"] > 0.05
    assert abs(lateral.summary["dy"]) > abs(lateral.summary["dx"]) + 0.03


def test_viewer_yaw_command_changes_yaw_and_touchdown_metrics(real_runtime):
    yaw_left = real_runtime.plan_case("yaw_left")

    assert yaw_left.summary["standstill"] is False
    assert yaw_left.summary["dyaw"] > 0.05
    assert yaw_left.left_touchdown_mean_y < -0.01
    assert yaw_left.right_touchdown_mean_y > 0.01


def test_viewer_playback_matches_reference_frame_numeric(real_runtime):
    forward = real_runtime.plan_case("forward")
    frame_idx = min(7, forward.result.num_frames - 1)

    readback = real_runtime.playback_sync_authoritative_readback(forward.result, frame_idx=frame_idx)

    torch.testing.assert_close(readback.root_pos_w, forward.result.root_pos_w[:, frame_idx], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(readback.joint_pos, forward.result.joint_angles[:, frame_idx], atol=1e-4, rtol=1e-4)


def test_viewer_standstill_has_no_single_leg_outlier(real_runtime):
    standstill = real_runtime.plan_case("standstill")

    assert standstill.touchdown_xy_delta_norms.max().item() < 1e-5
    assert (standstill.touchdown_xy_delta_norms.max() - standstill.touchdown_xy_delta_norms.min()).item() < 1e-5


def test_viewer_leg_order_matches_planner_contract(real_runtime):
    assert real_runtime.foot_names == LEG_ORDER


def test_viewer_semantic_diagnostics_ignore_invalid_hits_and_cover_valid_partition(real_runtime):
    forward = real_runtime.plan_case("forward")
    diagnostics = forward.semantic_diagnostics

    assert diagnostics["valid_sample_count"] > 0
    assert diagnostics["terrain_hit_count"] + diagnostics["small_hit_count"] + diagnostics["large_hit_count"] == diagnostics["valid_sample_count"]
    assert diagnostics["height_lift_max"] >= 0.0


def test_viewer_together_semantic_smoke_reports_required_obstacle_hits(real_semantic_together_runtime):
    forward = real_semantic_together_runtime.plan_case("forward")
    diagnostics = forward.semantic_diagnostics

    assert diagnostics["valid_sample_count"] > 0
    assert diagnostics["terrain_hit_count"] > 0
    assert diagnostics["height_lift_max"] > 0.05
    assert forward.summary["dx"] > 0.05


def test_viewer_together_runtime_plan_exposes_grounded_crossing_diagnostics(real_semantic_together_runtime):
    forward = real_semantic_together_runtime.plan_case("forward")
    grounded = forward.grounded_crossing

    assert grounded is not None
    assert grounded.mode.shape == (1,)
    assert grounded.status.shape == (1,)
    assert grounded.feasible.shape == (1,)
    assert grounded.safe_fallback.shape == (1,)
    assert grounded.selected_beta.shape == (1,)
    assert grounded.selected_route.shape == (1,)
    assert grounded.direction_id.shape == (1,)
    assert grounded.command_direction_violation.shape == (1,)
    assert grounded.cross_small_success.shape == (1,)
    assert grounded.body_min_clearance.shape == (1,)
    assert grounded.leg_min_clearance.shape == (1,)
    assert grounded.base_min_clearance_to_small.shape == (1,)
    assert grounded.per_leg_touchdown_on_small_count.shape == (1, 4)
    assert grounded.per_leg_foot_small_collision_count.shape == (1, 4)
    assert grounded.per_leg_min_clearance_to_small.shape == (1, 4)
    assert grounded.per_leg_touchdown_beyond_small_back_edge.shape == (1, 4)
    assert grounded.touchdown_ground_gap_by_leg.shape == (1, 4)
    assert grounded.touchdown_semantic_by_leg.shape == (1, 4)


def test_viewer_together_runtime_plan_grounded_crossing_summary_uses_required_metric_names(real_semantic_together_runtime):
    forward = real_semantic_together_runtime.plan_case("forward")
    grounded_summary = forward.grounded_crossing_summary

    assert grounded_summary is not None
    assert set(grounded_summary) == {
        "mode",
        "status",
        "feasible",
        "safe_fallback",
        "selected_beta",
        "selected_route",
        "direction_id",
        "state_mode",
        "small_strategy_outcome",
        "command_direction_violation",
        "cross_small_success",
        "body_min_clearance",
        "leg_min_clearance",
        "base_min_clearance_to_small",
        "per_leg_touchdown_on_small_count",
        "per_leg_foot_small_collision_count",
        "per_leg_min_clearance_to_small",
        "per_leg_touchdown_beyond_small_back_edge",
        "touchdown_ground_gap_by_leg",
        "touchdown_semantic_by_leg",
        "front_touchdown_ground_gap",
        "rear_touchdown_ground_gap",
        "touchdown_on_small_count",
        "front_foot_small_collision_count",
        "rear_foot_small_collision_count",
        "base_small_penetration_count",
        "base_path_crosses_small_flag",
    }
    assert isinstance(grounded_summary["mode"], int)
    assert isinstance(grounded_summary["status"], int)
    assert isinstance(grounded_summary["feasible"], bool)
    assert isinstance(grounded_summary["safe_fallback"], bool)
    assert isinstance(grounded_summary["selected_beta"], float)
    assert isinstance(grounded_summary["selected_route"], int)
    assert isinstance(grounded_summary["direction_id"], int)
    assert isinstance(grounded_summary["state_mode"], int)
    assert isinstance(grounded_summary["small_strategy_outcome"], int)
    assert isinstance(grounded_summary["command_direction_violation"], bool)
    assert isinstance(grounded_summary["cross_small_success"], bool)
    assert isinstance(grounded_summary["body_min_clearance"], float)
    assert isinstance(grounded_summary["leg_min_clearance"], float)
    assert isinstance(grounded_summary["base_min_clearance_to_small"], float)
    assert isinstance(grounded_summary["per_leg_touchdown_on_small_count"], tuple)
    assert len(grounded_summary["per_leg_touchdown_on_small_count"]) == 4
    assert isinstance(grounded_summary["per_leg_foot_small_collision_count"], tuple)
    assert len(grounded_summary["per_leg_foot_small_collision_count"]) == 4
    assert isinstance(grounded_summary["per_leg_min_clearance_to_small"], tuple)
    assert len(grounded_summary["per_leg_min_clearance_to_small"]) == 4
    assert isinstance(grounded_summary["per_leg_touchdown_beyond_small_back_edge"], tuple)
    assert len(grounded_summary["per_leg_touchdown_beyond_small_back_edge"]) == 4
    assert isinstance(grounded_summary["touchdown_ground_gap_by_leg"], tuple)
    assert len(grounded_summary["touchdown_ground_gap_by_leg"]) == 4
    assert isinstance(grounded_summary["touchdown_semantic_by_leg"], tuple)
    assert len(grounded_summary["touchdown_semantic_by_leg"]) == 4
    assert isinstance(grounded_summary["front_touchdown_ground_gap"], tuple)
    assert len(grounded_summary["front_touchdown_ground_gap"]) == 2
    assert isinstance(grounded_summary["rear_touchdown_ground_gap"], tuple)
    assert len(grounded_summary["rear_touchdown_ground_gap"]) == 2
    assert isinstance(grounded_summary["touchdown_on_small_count"], int)
    assert isinstance(grounded_summary["front_foot_small_collision_count"], int)
    assert isinstance(grounded_summary["rear_foot_small_collision_count"], int)
    assert isinstance(grounded_summary["base_small_penetration_count"], int)
    assert isinstance(grounded_summary["base_path_crosses_small_flag"], bool)


def test_viewer_together_targeted_s4_small_scan_reports_semantic_hits(real_semantic_together_runtime):
    diagnostics = real_semantic_together_runtime.semantic_scan_near_s4_anchor("small")

    assert diagnostics["valid_sample_count"] > 0
    assert diagnostics["small_hit_count"] > 0


def test_viewer_together_targeted_s4_large_scan_reports_semantic_hits(real_semantic_together_runtime):
    diagnostics = real_semantic_together_runtime.semantic_scan_near_s4_anchor("large")

    assert diagnostics["valid_sample_count"] > 0
    assert diagnostics["large_hit_count"] > 0


def test_compact_semantic_runtime_shape_pool_includes_capsule_and_cone(real_semantic_together_runtime):
    shape_kinds = real_semantic_together_runtime.compact_semantic_shape_kinds()

    assert "capsule" in shape_kinds
    assert "cone" in shape_kinds


def test_viewer_batched_runtime_smoke_preserves_parallel_path(real_batched_runtime):
    batched = real_batched_runtime.plan_batched_cases(["batched_forward", "batched_lateral_left"])

    assert batched.root_pos_w.shape[0] == 2
    assert abs(batched.path_deltas[0, 0].item()) > abs(batched.path_deltas[0, 1].item()) + 0.03
    assert abs(batched.path_deltas[1, 1].item()) > abs(batched.path_deltas[1, 0].item()) + 0.03


def test_r1_cruise_no_semantic_no_bypass(real_semantic_together_runtime):
    cruise = real_semantic_together_runtime.plan_case_near_semantic_stage(
        "S1",
        command_name="forward",
        longitudinal_offset_m=0.0,
    )
    grounded = cruise.grounded_crossing

    assert grounded is not None
    assert int(grounded.mode.item()) == T116_MODE_CRUISE
    assert int(grounded.selected_route.item()) == 0
    assert bool(grounded.feasible.item()) is True
    assert bool(grounded.safe_fallback.item()) is False
    assert bool(grounded.command_direction_violation.item()) is False


def test_r2_small_cross_runtime_four_leg_success_all_command_directions(real_semantic_together_runtime):
    reports = real_semantic_together_runtime.grounded_crossing_runtime_sequences_by_command()

    assert set(reports) == {"forward", "backward", "lateral_left", "lateral_right"}
    for report in reports.values():
        assert T116_MODE_CROSS_SMALL in report.mode_sequence
        assert all(report.feasible_sequence)
        assert report.cross_small_success_count > 0
        assert report.command_direction_violation_count == 0
        assert report.cross_outcome_grounded == 1


def test_r3_small_cross_runtime_no_touchdown_on_small_all_directions(real_semantic_together_runtime):
    reports = real_semantic_together_runtime.grounded_crossing_runtime_sequences_by_command()

    for report in reports.values():
        assert report.touchdown_on_small_count == 0
        assert report.per_leg_touchdown_on_small_count == (0, 0, 0, 0)
        assert report.touchdown_semantic_by_leg == (0, 0, 0, 0)


def test_r4_small_cross_runtime_no_foot_path_collision_all_directions(real_semantic_together_runtime):
    reports = real_semantic_together_runtime.grounded_crossing_runtime_sequences_by_command()

    for report in reports.values():
        assert report.foot_small_collision_count == 0
        assert report.per_leg_foot_small_collision_count == (0, 0, 0, 0)
        assert min(report.per_leg_min_clearance_to_small_m) >= 0.0


def test_r5_small_cross_runtime_no_base_body_leg_penetration_all_directions(real_semantic_together_runtime):
    reports = real_semantic_together_runtime.grounded_crossing_runtime_sequences_by_command()

    for report in reports.values():
        assert report.base_small_penetration_count == 0
        assert report.base_path_crosses_small_flag == 0
        assert report.base_min_clearance_to_small_m >= 0.0
        assert report.body_min_clearance_m >= 0.0
        assert report.leg_min_clearance_m >= 0.0


def test_r6_large_runtime_bypass_direction_guard(real_semantic_together_runtime):
    reports = real_semantic_together_runtime.grounded_crossing_runtime_sequences_by_command(semantic_class="large")

    for report in reports.values():
        assert report.status_sequence
        assert all(status == int(TogetherPlannerStatus.OK) for status in report.status_sequence)
        assert all(report.feasible_sequence)
        assert not any(report.safe_fallback_sequence)
        assert T116_MODE_BYPASS_OBSTACLE in report.mode_sequence
        assert report.command_direction_violation_count == 0
        assert any(route != 0 for route in report.selected_route_sequence)
        assert report.touchdown_on_small_count == 0
        assert report.touchdown_semantic_by_leg == (0, 0, 0, 0)
        assert report.foot_small_collision_count == 0
        assert report.base_small_penetration_count == 0
        assert report.base_path_crosses_small_flag == 0
        assert report.base_min_clearance_to_small_m >= 0.0
        assert report.body_min_clearance_m >= 0.0
        assert report.leg_min_clearance_m >= 0.0


def test_r7_lateral_runtime_no_opposite_direction_rejection_left_and_right(real_semantic_together_runtime):
    reports = real_semantic_together_runtime.grounded_crossing_runtime_sequences_by_command(
        command_names=("lateral_left", "lateral_right"),
    )
    left = reports["lateral_left"]
    right = reports["lateral_right"]

    assert left.command_direction_violation_count == 0
    assert right.command_direction_violation_count == 0
    assert set(left.direction_id_sequence) == {DIRECTION_LATERAL_LEFT}
    assert set(right.direction_id_sequence) == {DIRECTION_LATERAL_RIGHT}
    assert all(beta >= 0.0 for beta in left.selected_beta_sequence)
    assert all(beta >= 0.0 for beta in right.selected_beta_sequence)
