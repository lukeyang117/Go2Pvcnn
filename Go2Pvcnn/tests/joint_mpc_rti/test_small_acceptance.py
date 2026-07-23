from __future__ import annotations

from dataclasses import replace
import inspect
import math
from types import SimpleNamespace

import pytest
import torch


def _metric(name: str, value: float, *, passed: bool = True):
    from .joint_metrics import MetricResult

    return MetricResult(
        name=name,
        value=value,
        numerator=value,
        denominator=1,
        valid_count=1,
        applicable=True,
        na_reason=None,
        threshold=0.0,
        passed=passed,
        worst_case_key=("small",),
    )


def _passing_small_report():
    from .joint_metrics import applicable_metrics
    from .run_joint_acceptance import AcceptanceCell, AcceptanceReport, CellReport

    cell = AcceptanceCell(
        scenario="small",
        command=(0.2, 0.0, 0.0),
        phase=0,
        shape="cuboid",
        offset=0.0,
    )
    metrics = {name: _metric(name, 0.0) for name in applicable_metrics("small", cell.command)}
    return AcceptanceReport(
        stage="small",
        code_ref="test",
        cells=(CellReport(cell=cell, metrics=metrics, passed=True),),
    )


def test_small_gate_requires_all_flat_metrics_and_small_metrics() -> None:
    from .run_joint_acceptance import require_small_gate

    report = _passing_small_report()
    cell_report = report.cells[0]
    metrics = dict(cell_report.metrics)
    metrics["stance_ground_gap"] = replace(metrics["stance_ground_gap"], passed=False)
    failed = replace(
        report,
        cells=(replace(cell_report, metrics=metrics, passed=False),),
    )

    assert not require_small_gate(failed).passed
    assert require_small_gate(report).passed


def test_small_fixture_uses_real_scanner_resolution() -> None:
    from .run_joint_acceptance import build_small_obstacle_field

    resolution = inspect.signature(build_small_obstacle_field).parameters["resolution"].default

    assert resolution == pytest.approx(0.01)


def test_bypass_or_stop_is_not_strict_crossing_success() -> None:
    from .run_joint_acceptance import strict_crossing_event

    crossing = torch.tensor(
        [[[-0.40, 0.00], [-0.15, 0.00], [0.15, 0.00], [0.40, 0.00]]]
    )
    bypass = crossing.clone()
    bypass[:, 1:3, 1] = 0.45
    stopped = crossing.clone()
    stopped[:, -1, 0] = 0.10
    command = torch.tensor([[1.0, 0.0]])
    center = torch.zeros(1, 2)

    assert strict_crossing_event(crossing, command, center, radius_m=0.10).success.item()
    assert not strict_crossing_event(bypass, command, center, radius_m=0.10).success.item()
    assert not strict_crossing_event(stopped, command, center, radius_m=0.10).success.item()


def test_offset_small_obstacle_intersects_swept_robot_footprint() -> None:
    from .run_joint_acceptance import strict_crossing_event

    root = torch.tensor([[[-0.40, 0.00], [-0.10, 0.00], [0.10, 0.00], [0.40, 0.00]]])
    command = torch.tensor([[1.0, 0.0]])
    center = torch.tensor([[0.0, 0.24]])

    result = strict_crossing_event(root, command, center, radius_m=0.06)

    assert result.success.item()
    assert result.intersected_footprint.item()


def _strict_foot_cross_inputs(*, moving_root: bool = True):
    root = torch.tensor(
        [[[-0.30, 0.0], [-0.10, 0.0], [0.10, 0.0], [0.30, 0.0]]]
    )
    if not moving_root:
        root.zero_()
    foot = torch.zeros((1, 4, 4, 3))
    foot[..., 2] = 0.022
    foot[:, :, 0, 0] = torch.tensor((-0.20, -0.10, 0.10, 0.20))
    foot[:, 1:3, 0, 2] = 0.25
    contact = torch.ones((1, 4, 4), dtype=torch.bool)
    contact[:, 1:3, 0] = False
    collision = {
        part: torch.zeros((1, 4), dtype=torch.bool)
        for part in ("foot", "knee", "calf", "thigh", "base")
    }
    return root, foot, contact, collision


def test_strict_cross_requires_swept_sole_height_landing_and_body_safety() -> None:
    from .run_joint_acceptance import strict_crossing_event

    root, foot, contact, collision = _strict_foot_cross_inputs()
    result = strict_crossing_event(
        root,
        torch.tensor([[0.2, 0.0]]),
        torch.zeros((1, 2)),
        radius_m=0.06,
        foot_pos_w=foot,
        contact_state=contact,
        obstacle_top_z=torch.tensor([0.16]),
        part_collision=collision,
        landing_safe=torch.ones((1, 4), dtype=torch.bool),
        dt=0.02,
    )

    assert result.opportunity.item()
    assert result.over_xy.item()
    assert result.over_z.item()
    assert result.direction_ok.item()
    assert result.after.item()
    assert result.land_ok.item()
    assert result.body_ok.item()
    assert result.success.item()

    collision["calf"][0, 2] = True
    unsafe = strict_crossing_event(
        root,
        torch.tensor([[0.2, 0.0]]),
        torch.zeros((1, 2)),
        radius_m=0.06,
        foot_pos_w=foot,
        contact_state=contact,
        obstacle_top_z=torch.tensor([0.16]),
        part_collision=collision,
        landing_safe=torch.ones((1, 4), dtype=torch.bool),
        dt=0.02,
    )
    assert not unsafe.body_ok.item()
    assert not unsafe.success.item()


def test_cross_opportunity_with_no_actual_root_progress_is_failure_not_na() -> None:
    from .run_joint_acceptance import strict_crossing_event

    root, foot, contact, collision = _strict_foot_cross_inputs(moving_root=False)
    result = strict_crossing_event(
        root,
        torch.tensor([[0.2, 0.0]]),
        torch.zeros((1, 2)),
        radius_m=0.06,
        foot_pos_w=foot,
        contact_state=contact,
        obstacle_top_z=torch.tensor([0.16]),
        part_collision=collision,
        landing_safe=torch.ones((1, 4), dtype=torch.bool),
        dt=0.02,
    )

    assert result.opportunity.item()
    assert not result.direction_applicable.item()
    assert not result.success.item()


def test_zero_translation_and_pure_yaw_mark_crossing_not_applicable() -> None:
    from .joint_metrics import applicable_metrics

    assert "strict_cross_success" not in applicable_metrics("small", (0.0, 0.0, 0.0))
    assert "strict_cross_success" not in applicable_metrics("small", (0.0, 0.0, 1.0))
    assert "strict_cross_success" in applicable_metrics("small", (0.2, 0.0, 0.0))


def test_small_field_preserves_native_shape_top_and_semantic_footprint() -> None:
    from extension.joint_mpc_rti.terrain.query import query_world
    from .run_joint_acceptance import build_small_obstacle_field
    from .scenario_matrix import SMALL_SHAPES

    for shape in SMALL_SHAPES:
        field, center, radius = build_small_obstacle_field(
            commands=torch.tensor([[0.2, 0.0, 0.0]]),
            shapes=(shape,),
            offsets=torch.tensor([0.0]),
            device="cpu",
        )
        points = torch.stack(
            (
                center,
                center + torch.tensor([[radius + 0.04, 0.0]]),
            ),
            dim=1,
        )
        query = query_world(field, points)

        assert query.height_w[0, 0] >= 0.12
        assert query.semantic_id[0, 0].item() == 1
        assert query.height_w[0, 1].item() == 0.0
        assert query.semantic_id[0, 1].item() == 0


@pytest.mark.parametrize(
    ("shape", "expected_center", "expected_half_radius", "expected_edge"),
    (
        ("sphere", 0.12, 0.06 + math.sqrt(0.06**2 - 0.03**2), 0.06),
        ("cuboid", 0.16, 0.16, 0.16),
        ("cylinder", 0.16, 0.16, 0.16),
        ("capsule", 0.16, 0.10 + math.sqrt(0.06**2 - 0.03**2), 0.10),
        ("cone", 0.16, 0.08, 0.0),
    ),
)
def test_small_field_matches_grounded_native_shape_height_profile(
    shape: str,
    expected_center: float,
    expected_half_radius: float,
    expected_edge: float,
) -> None:
    from extension.joint_mpc_rti.terrain.query import query_world
    from .run_joint_acceptance import build_small_obstacle_field

    field, center, radius = build_small_obstacle_field(
        commands=torch.tensor([[0.2, 0.0, 0.0]]),
        shapes=(shape,),
        offsets=torch.tensor([0.0]),
        device="cpu",
    )
    points = torch.stack(
        (
            center,
            center + torch.tensor([[0.5 * radius, 0.0]]),
            center + torch.tensor([[radius, 0.0]]),
        ),
        dim=1,
    )
    query = query_world(field, points)

    torch.testing.assert_close(
        query.height_w[0],
        torch.tensor((expected_center, expected_half_radius, expected_edge)),
        atol=1.0e-4,
        rtol=0.0,
    )
    assert query.semantic_id[0].tolist() == [1, 1, 1]


def test_small_field_center_is_relative_to_current_root_origin() -> None:
    from .run_joint_acceptance import build_small_obstacle_field

    origin = torch.tensor([[1.2, -0.7]])
    field, center, _ = build_small_obstacle_field(
        commands=torch.tensor([[0.2, 0.0, 0.0]]),
        shapes=("sphere",),
        offsets=torch.tensor([0.0]),
        origin_xy_w=origin,
        device="cpu",
    )

    torch.testing.assert_close(field.origin_w[:, :2], origin)
    torch.testing.assert_close(center, origin + torch.tensor([[0.40, 0.0]]))


def test_small_field_recenters_on_root_without_moving_world_obstacle() -> None:
    from extension.joint_mpc_rti.terrain.query import query_world
    from .run_joint_acceptance import build_small_obstacle_field

    command = torch.tensor([[0.2, 0.0, 0.0]])
    _, obstacle_center, _ = build_small_obstacle_field(
        commands=command,
        shapes=("sphere",),
        offsets=torch.tensor([0.0]),
        device="cpu",
    )
    root_xy = torch.tensor([[0.2, 0.0]])

    field, center, _ = build_small_obstacle_field(
        commands=command,
        shapes=("sphere",),
        offsets=torch.tensor([0.0]),
        origin_xy_w=root_xy,
        obstacle_center_xy_w=obstacle_center,
        device="cpu",
    )

    torch.testing.assert_close(field.origin_w[:, :2], root_xy)
    torch.testing.assert_close(center, obstacle_center)
    assert query_world(field, obstacle_center[:, None]).semantic_id.item() == 1


def test_small_field_uses_exact_semantic_occupancy() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiTerrainCfg
    from extension.joint_mpc_rti.terrain.query import query_world
    from .run_joint_acceptance import build_small_obstacle_field

    field, center, _ = build_small_obstacle_field(
        commands=torch.tensor([[0.2, 0.0, 0.0]]),
        shapes=("sphere",),
        offsets=torch.tensor([0.0]),
        terrain_cfg=JointMpcRtiTerrainCfg(),
        device="cpu",
    )

    assert query_world(field, center[:, None]).small_occupancy.item() == 1.0


def test_small_obstacle_starts_ahead_of_the_robot_footprint() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.terrain.query import query_world
    from .helpers import make_state
    from .run_joint_acceptance import build_small_obstacle_field

    measured = make_state(1)
    feet = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos).foot_pos_w
    field, center, _ = build_small_obstacle_field(
        commands=torch.tensor([[0.0, 0.0, 0.0]]),
        shapes=("sphere",),
        offsets=torch.tensor([0.0]),
        device="cpu",
    )

    assert center[0, 0] >= 0.35
    assert (query_world(field, feet).small_distance_m > 0.0).all()


def test_small_trace_emits_detector_values_instead_of_flat_placeholders() -> None:
    from .run_joint_acceptance import AcceptanceCell, simulate_small_trace

    cell = AcceptanceCell(
        scenario="small",
        command=(0.2, 0.0, 0.0),
        phase=0,
        shape="sphere",
        offset=0.0,
    )
    trace = simulate_small_trace((cell,), steps=2, device="cpu")

    assert trace.strict_cross_success is not None
    assert trace.touchdown_on_small is not None
    assert trace.stance_on_small is not None
    assert trace.airborne_touchdown is not None
    assert trace.part_penetration_m is not None
    assert set(trace.part_collision) == {"foot", "knee", "calf", "thigh", "base"}
    assert not torch.all(trace.foot_small_distance_m == 1.0)


def test_nonzero_phase_small_trace_enters_formal_window_warm_only() -> None:
    from .run_joint_acceptance import AcceptanceCell, simulate_small_trace

    cell = AcceptanceCell(
        scenario="small",
        command=(0.0, 0.0, 0.0),
        phase=5,
        shape="sphere",
        offset=0.0,
    )

    trace = simulate_small_trace((cell,), steps=2, device="cpu")

    assert trace.cold_start.sum().item() == 1
    assert trace.warm_start[:, 6:].all()
    assert trace.gait_phase[0, 5].item() == 5


def test_small_acceptance_runner_returns_small_report() -> None:
    from .run_joint_acceptance import AcceptanceCell, run_small_acceptance

    cell = AcceptanceCell(
        scenario="small",
        command=(0.0, 0.0, 0.0),
        phase=0,
        shape="sphere",
        offset=0.0,
    )
    report = run_small_acceptance(cells=(cell,), steps=1, device="cpu")

    assert report.stage == "small"
    assert report.cells[0].cell == cell
    assert "foot_collision_frame_rate" in report.cells[0].metrics


def test_small_acceptance_groups_mixed_phases_and_preserves_cell_order(monkeypatch) -> None:
    from . import joint_metrics, run_joint_acceptance
    from .run_joint_acceptance import AcceptanceCell

    cells = tuple(
        AcceptanceCell(
            scenario="small",
            command=(0.2, 0.0, 0.0),
            phase=phase,
            shape="sphere",
            offset=float(index) * 0.01,
        )
        for index, phase in enumerate((5, 2, 5, 2))
    )
    simulated_groups = []

    def fake_simulate(group, *, steps, device):
        assert len({cell.phase for cell in group}) == 1
        simulated_groups.append(group)
        return group

    monkeypatch.setattr(run_joint_acceptance, "simulate_small_trace", fake_simulate)
    monkeypatch.setattr(run_joint_acceptance, "_slice_trace", lambda trace, index: trace[index])
    monkeypatch.setattr(
        joint_metrics,
        "evaluate_trace",
        lambda cell, **_: SimpleNamespace(metrics={}, passed=True),
    )

    report = run_joint_acceptance.run_small_acceptance(
        cells=cells,
        steps=1,
        device="cpu",
        cell_batch_size=4,
    )

    assert tuple(tuple(cell.phase for cell in group) for group in simulated_groups) == ((5, 5), (2, 2))
    assert tuple(item.cell for item in report.cells) == cells
