"""Unified flat/small acceptance entrypoint and report schema."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (REPO_ROOT, REPO_ROOT / "Go2Pvcnn"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from .joint_metrics import MetricResult, applicable_metrics
    from .scenario_matrix import COMMANDS, SMALL_OFFSETS, SMALL_PHASES, SMALL_SHAPES
except ImportError:  # pragma: no cover - direct script execution
    from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import MetricResult, applicable_metrics
    from Go2Pvcnn.tests.joint_mpc_rti.scenario_matrix import COMMANDS, SMALL_OFFSETS, SMALL_PHASES, SMALL_SHAPES


RANKED_COMMANDS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, -0.5, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)

SMALL_DIAMETER_M = 0.12
SMALL_HEIGHT_M = 0.16
SMALL_CENTER_PROGRESS_M = 0.40
ROBOT_FOOTPRINT_HALF_WIDTH_M = 0.24


def _grounded_native_small_height_profile(shape: str, radial: torch.Tensor, radius: float) -> torch.Tensor:
    spherical_cap = torch.sqrt((float(radius) ** 2 - radial.square()).clamp_min(0.0))
    if shape == "sphere":
        return float(radius) + spherical_cap
    if shape in {"cuboid", "cylinder"}:
        return torch.full_like(radial, SMALL_HEIGHT_M)
    if shape == "capsule":
        cylinder_height = max(SMALL_HEIGHT_M - SMALL_DIAMETER_M, 1.0e-6)
        return float(radius) + cylinder_height + spherical_cap
    if shape == "cone":
        return SMALL_HEIGHT_M * (1.0 - radial / float(radius)).clamp_min(0.0)
    raise ValueError(f"unsupported small shape: {shape}")


@dataclass(frozen=True)
class AcceptanceCell:
    scenario: str
    command: tuple[float, float, float]
    phase: int | None = None
    shape: str | None = None
    offset: float | None = None
    environment: int = 0

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.scenario,
            *(f"{value:.6g}" for value in self.command),
            "na" if self.phase is None else str(self.phase),
            self.shape or "na",
            "na" if self.offset is None else f"{self.offset:.6g}",
            str(self.environment),
        )


@dataclass(frozen=True)
class CellReport:
    cell: AcceptanceCell
    metrics: dict[str, MetricResult]
    passed: bool


@dataclass(frozen=True)
class AcceptanceReport:
    stage: str
    code_ref: str
    cells: tuple[CellReport, ...]

    @property
    def passed(self) -> bool:
        return bool(self.cells) and all(cell.passed for cell in self.cells)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "code_ref": self.code_ref,
            "passed": self.passed,
            "cells": [asdict(cell) for cell in self.cells],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)


@dataclass(frozen=True)
class ShardMetadata:
    index: int
    count: int
    total_cells: int
    selected_cells: int


@dataclass(frozen=True)
class GateResult:
    stage: str
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class StrictCrossingResult:
    success: torch.Tensor
    crossed_longitudinal: torch.Tensor
    intersected_footprint: torch.Tensor


def acceptance_report_from_dict(payload: Mapping[str, object]) -> AcceptanceReport:
    cell_reports: list[CellReport] = []
    for item in payload.get("cells", []):
        cell_payload = item["cell"]
        cell = AcceptanceCell(
            scenario=str(cell_payload["scenario"]),
            command=tuple(float(value) for value in cell_payload["command"]),
            phase=None if cell_payload.get("phase") is None else int(cell_payload["phase"]),
            shape=None if cell_payload.get("shape") is None else str(cell_payload["shape"]),
            offset=None if cell_payload.get("offset") is None else float(cell_payload["offset"]),
            environment=int(cell_payload.get("environment", 0)),
        )
        metrics: dict[str, MetricResult] = {}
        for name, metric_payload in item.get("metrics", {}).items():
            worst_case_key = metric_payload.get("worst_case_key")
            metrics[str(name)] = MetricResult(
                name=str(metric_payload["name"]),
                value=metric_payload.get("value"),
                numerator=metric_payload.get("numerator"),
                denominator=metric_payload.get("denominator"),
                valid_count=int(metric_payload["valid_count"]),
                applicable=bool(metric_payload["applicable"]),
                na_reason=metric_payload.get("na_reason"),
                threshold=metric_payload.get("threshold"),
                passed=metric_payload.get("passed"),
                worst_case_key=None if worst_case_key is None else tuple(map(str, worst_case_key)),
            )
        cell_reports.append(
            CellReport(cell=cell, metrics=metrics, passed=bool(item["passed"]))
        )
    return AcceptanceReport(
        stage=str(payload["stage"]),
        code_ref=str(payload["code_ref"]),
        cells=tuple(cell_reports),
    )


def select_cell_shard(
    cells: tuple[AcceptanceCell, ...],
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[AcceptanceCell, ...]:
    count = int(shard_count)
    index = int(shard_index)
    if count <= 0:
        raise ValueError("shard_count must be positive")
    if index < 0 or index >= count:
        raise ValueError("shard_index must be in [0, shard_count)")
    start = len(cells) * index // count
    stop = len(cells) * (index + 1) // count
    return cells[start:stop]


def _shard_metadata_from_payload(payload: Mapping[str, object]) -> ShardMetadata:
    shard = payload.get("shard")
    if not isinstance(shard, Mapping):
        raise ValueError("every shard report must contain shard metadata")
    return ShardMetadata(
        index=int(shard["index"]),
        count=int(shard["count"]),
        total_cells=int(shard["total_cells"]),
        selected_cells=int(shard["selected_cells"]),
    )


def merge_acceptance_shard_payloads(
    payloads: Sequence[Mapping[str, object]],
    *,
    expected_cells: tuple[AcceptanceCell, ...],
    stage: str,
) -> AcceptanceReport:
    if not payloads:
        raise ValueError("at least one shard report is required")
    metadata = tuple(_shard_metadata_from_payload(payload) for payload in payloads)
    counts = {item.count for item in metadata}
    if len(counts) != 1:
        raise ValueError("all shard reports must use the same shard count")
    shard_count = counts.pop()
    indices = [item.index for item in metadata]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate shard indices")
    missing = sorted(set(range(shard_count)) - set(indices))
    if missing:
        raise ValueError(f"missing shard indices: {missing}")
    if sorted(indices) != list(range(shard_count)):
        raise ValueError("shard indices must exactly cover [0, shard_count)")

    reports = tuple(acceptance_report_from_dict(payload) for payload in payloads)
    if {report.stage for report in reports} != {stage}:
        raise ValueError(f"all shard reports must use stage {stage!r}")
    if len({report.code_ref for report in reports}) != 1:
        raise ValueError("all shard reports must use the same code_ref")
    if len({cell.key for cell in expected_cells}) != len(expected_cells):
        raise ValueError("expected cells contain duplicate keys")

    reports_by_index = {
        item.index: report for item, report in zip(metadata, reports, strict=True)
    }
    metadata_by_index = {item.index: item for item in metadata}
    report_by_key: dict[tuple[str, ...], CellReport] = {}
    for index in range(shard_count):
        expected_shard = select_cell_shard(
            expected_cells,
            shard_count=shard_count,
            shard_index=index,
        )
        report = reports_by_index[index]
        item = metadata_by_index[index]
        if item.total_cells != len(expected_cells) or item.selected_cells != len(expected_shard):
            raise ValueError("shard cell counts do not match the expected matrix")
        if tuple(cell.cell.key for cell in report.cells) != tuple(cell.key for cell in expected_shard):
            raise ValueError(f"shard {index} cell keys do not match the expected partition")
        for cell_report in report.cells:
            if cell_report.cell.key in report_by_key:
                raise ValueError(f"duplicate cell key: {cell_report.cell.key}")
            report_by_key[cell_report.cell.key] = cell_report

    if set(report_by_key) != {cell.key for cell in expected_cells}:
        raise ValueError("merged shard cell keys do not match the expected matrix")
    return AcceptanceReport(
        stage=stage,
        code_ref=reports[0].code_ref,
        cells=tuple(report_by_key[cell.key] for cell in expected_cells),
    )


def require_gate(report: AcceptanceReport, *, stage: str) -> GateResult:
    failures: list[str] = []
    if report.stage != stage:
        failures.append(f"report stage is {report.stage!r}, expected {stage!r}")
    for cell_report in report.cells:
        prefix = "/".join(cell_report.cell.key)
        for name, metric in cell_report.metrics.items():
            if metric.applicable and metric.passed is not True:
                failures.append(f"{prefix}:{name}")
    if not report.cells:
        failures.append("report has no cells")
    return GateResult(stage=stage, passed=not failures, failures=tuple(failures))


def require_flat_gate(report: AcceptanceReport) -> GateResult:
    return require_gate(report, stage="flat")


def require_small_gate(report: AcceptanceReport) -> GateResult:
    return require_gate(report, stage="small")


def strict_crossing_event(
    root_xy_w: torch.Tensor,
    command_xy_w: torch.Tensor,
    obstacle_center_xy_w: torch.Tensor,
    *,
    radius_m: float,
    footprint_half_width_m: float = ROBOT_FOOTPRINT_HALF_WIDTH_M,
) -> StrictCrossingResult:
    root = torch.as_tensor(root_xy_w, dtype=torch.float32)
    command = torch.as_tensor(command_xy_w, dtype=root.dtype, device=root.device)
    center = torch.as_tensor(obstacle_center_xy_w, dtype=root.dtype, device=root.device)
    if root.ndim != 3 or root.shape[-1] != 2:
        raise ValueError("root_xy_w must have shape [B,T,2]")
    if command.shape != (root.shape[0], 2) or center.shape != (root.shape[0], 2):
        raise ValueError("command and obstacle center must have shape [B,2]")
    direction = command / torch.linalg.vector_norm(command, dim=-1, keepdim=True).clamp_min(1.0e-6)
    lateral_axis = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    relative = root - center[:, None]
    longitudinal = (relative * direction[:, None]).sum(dim=-1)
    lateral = (relative * lateral_axis[:, None]).sum(dim=-1)
    radius = float(radius_m)
    lateral_limit = radius + float(footprint_half_width_m)
    crossed = (longitudinal[:, 0] < -radius) & (longitudinal[:, -1] > radius)
    in_longitudinal_band = longitudinal.abs() <= radius
    sampled_intersection = (in_longitudinal_band & (lateral.abs() <= lateral_limit)).any(dim=1)
    segment_crosses_center = (longitudinal[:, :-1] <= 0.0) & (longitudinal[:, 1:] >= 0.0)
    delta = longitudinal[:, 1:] - longitudinal[:, :-1]
    fraction = (-longitudinal[:, :-1] / delta.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    crossing_lateral = lateral[:, :-1] + fraction * (lateral[:, 1:] - lateral[:, :-1])
    segment_intersection = (
        segment_crosses_center & (crossing_lateral.abs() <= lateral_limit)
    ).any(dim=1)
    intersects = sampled_intersection | segment_intersection
    command_active = torch.linalg.vector_norm(command, dim=-1) > 1.0e-6
    return StrictCrossingResult(
        success=crossed & intersects & command_active,
        crossed_longitudinal=crossed,
        intersected_footprint=intersects,
    )


def build_small_obstacle_field(
    *,
    commands: torch.Tensor,
    shapes: tuple[str, ...],
    offsets: torch.Tensor,
    device: str,
    resolution: float = 0.01,
    origin_xy_w: torch.Tensor | None = None,
    obstacle_center_xy_w: torch.Tensor | None = None,
    terrain_cfg=None,
):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    command = torch.as_tensor(commands, dtype=torch.float32, device=device)
    offset = torch.as_tensor(offsets, dtype=command.dtype, device=command.device)
    batch = int(command.shape[0])
    if command.shape != (batch, 3) or len(shapes) != batch or offset.shape != (batch,):
        raise ValueError("commands, shapes, and offsets must share batch size")
    origin_xy = (
        torch.zeros(batch, 2, dtype=command.dtype, device=command.device)
        if origin_xy_w is None
        else torch.as_tensor(origin_xy_w, dtype=command.dtype, device=command.device)
    )
    if origin_xy.shape != (batch, 2):
        raise ValueError("origin_xy_w must have shape [B,2]")
    direction = command[:, :2] / torch.linalg.vector_norm(command[:, :2], dim=-1, keepdim=True).clamp_min(1.0e-6)
    fallback = torch.tensor((1.0, 0.0), dtype=command.dtype, device=command.device).expand(batch, -1)
    direction = torch.where(
        (torch.linalg.vector_norm(command[:, :2], dim=-1) > 1.0e-6)[:, None],
        direction,
        fallback,
    )
    lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    if obstacle_center_xy_w is None:
        local_center = direction * SMALL_CENTER_PROGRESS_M + lateral * offset[:, None]
        center = origin_xy + local_center
    else:
        center = torch.as_tensor(
            obstacle_center_xy_w, dtype=command.dtype, device=command.device
        )
        if center.shape != (batch, 2):
            raise ValueError("obstacle_center_xy_w must have shape [B,2]")
        local_center = center - origin_xy
    side = 151
    coordinate = (torch.arange(side, dtype=command.dtype, device=command.device) - (side - 1) / 2) * float(resolution)
    grid_x, grid_y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    relative_x = grid_x[None] - local_center[:, 0, None, None]
    relative_y = grid_y[None] - local_center[:, 1, None, None]
    radial = torch.sqrt(relative_x.square() + relative_y.square())
    radius = 0.5 * SMALL_DIAMETER_M
    height = torch.zeros(batch, side, side, dtype=command.dtype, device=command.device)
    semantic = torch.zeros(batch, side, side, dtype=torch.long, device=command.device)
    for index, shape in enumerate(shapes):
        if shape == "cuboid":
            mask = (relative_x[index].abs() <= radius) & (relative_y[index].abs() <= radius)
        elif shape in {"sphere", "cylinder", "capsule", "cone"}:
            mask = radial[index] <= radius
        else:
            raise ValueError(f"unsupported small shape: {shape}")
        profile = _grounded_native_small_height_profile(shape, radial[index], radius)
        height[index] = torch.where(mask, profile, height[index])
        semantic[index] = torch.where(mask, torch.ones_like(semantic[index]), semantic[index])
    field = build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=torch.cat(
            (origin_xy, torch.zeros(batch, 1, dtype=command.dtype, device=command.device)),
            dim=-1,
        ),
        yaw_w=torch.zeros(batch, dtype=command.dtype, device=command.device),
        timestamp=torch.zeros(batch, dtype=command.dtype, device=command.device),
        version=torch.ones(batch, dtype=torch.long, device=command.device),
        resolution=resolution,
        small_ids=(1,),
        large_ids=(2,),
        terrain_cfg=terrain_cfg,
    )
    return field, center, radius


def _slice_trace(trace, index: int):
    try:
        from .joint_metrics import JointMetricTrace
    except ImportError:  # pragma: no cover - direct script execution
        from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import JointMetricTrace

    def pick(value):
        return None if value is None else value[index : index + 1]

    return JointMetricTrace(
        root_pos_w=trace.root_pos_w[index : index + 1],
        root_rpy_w=trace.root_rpy_w[index : index + 1],
        joint_pos=trace.joint_pos[index : index + 1],
        foot_pos_w=trace.foot_pos_w[index : index + 1],
        contact_state=trace.contact_state[index : index + 1],
        command_body=trace.command_body[index : index + 1],
        gait_phase=trace.gait_phase[index : index + 1],
        foot_height_w=trace.foot_height_w[index : index + 1],
        foot_small_distance_m=trace.foot_small_distance_m[index : index + 1],
        part_collision={name: value[index : index + 1] for name, value in trace.part_collision.items()},
        line_alpha=trace.line_alpha[index : index + 1],
        nominal_root_pos_w=trace.nominal_root_pos_w[index : index + 1],
        nominal_root_rpy_w=trace.nominal_root_rpy_w[index : index + 1],
        valid=trace.valid[index : index + 1],
        map_valid=trace.map_valid[index : index + 1],
        timestamps=trace.timestamps[index : index + 1],
        dt=trace.dt,
        stance_anchor_w=pick(trace.stance_anchor_w),
        strict_cross_success=pick(trace.strict_cross_success),
        touchdown_on_small=pick(trace.touchdown_on_small),
        stance_on_small=pick(trace.stance_on_small),
        airborne_touchdown=pick(trace.airborne_touchdown),
        part_penetration_m=None
        if trace.part_penetration_m is None
        else {name: value[index : index + 1] for name, value in trace.part_penetration_m.items()},
        x0_injection_error=pick(trace.x0_injection_error),
        published_x1_error=pick(trace.published_x1_error),
        warm_start_jump=pick(trace.warm_start_jump),
        cold_start=pick(trace.cold_start),
        warm_start=pick(trace.warm_start),
        warm_cache_invariant_fault=pick(trace.warm_cache_invariant_fault),
    )


def simulate_flat_trace(
    commands: torch.Tensor,
    *,
    steps: int,
    device: str = "cpu",
    cfg=None,
):
    """Run the pure-kinematic planner in batch and return one shared flat trace."""
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.planner import step as planner_step
    from extension.joint_mpc_rti.terrain.query import query_world
    from extension.joint_mpc_rti.types import JointMpcRtiState
    try:
        from .helpers import make_state
    except ImportError:  # pragma: no cover - direct script execution
        from Go2Pvcnn.tests.joint_mpc_rti.helpers import make_state
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    commands = torch.as_tensor(commands, dtype=torch.float32, device=device)
    batch = int(commands.shape[0])
    measured = make_state(batch, device=device)
    base_field = build_field_batch(
        height_w=torch.zeros(batch, 151, 151, device=device),
        semantic_id=torch.zeros(batch, 151, 151, dtype=torch.long, device=device),
        origin_w=torch.zeros(batch, 3, device=device),
        yaw_w=torch.zeros(batch, device=device),
        timestamp=torch.zeros(batch, device=device),
        version=torch.ones(batch, dtype=torch.long, device=device),
        resolution=0.02,
        small_ids=(1,),
        large_ids=(2,),
    )
    cfg = JointMpcRtiCfg() if cfg is None else cfg
    solver_state = None
    state_rows = [measured.as_vector()]
    root_rpy_rows = [measured.root_rpy_w]
    joint_rows = [measured.joint_pos]
    foot_rows = []
    contact_rows = []
    phase_rows = [torch.zeros(batch, dtype=torch.long, device=device)]
    alpha_rows = [torch.ones(batch, device=device)]
    valid_rows = [torch.ones(batch, dtype=torch.bool, device=device)]
    map_valid_rows = [torch.ones(batch, dtype=torch.bool, device=device)]
    x0_errors = [torch.zeros(batch, device=device)]
    x1_errors = [torch.zeros(batch, device=device)]
    cold_start_rows = [torch.zeros(batch, dtype=torch.bool, device=device)]
    warm_start_rows = [torch.zeros(batch, dtype=torch.bool, device=device)]
    timestamps = [torch.zeros(batch, device=device)]
    initial_geometry = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos)
    foot_rows.append(initial_geometry.foot_pos_w)

    for step_index in range(int(steps)):
        field_origin = base_field.origin_w.clone()
        field_origin[:, :2] = measured.root_pos_w[:, :2]
        field = replace(base_field, origin_w=field_origin)
        measured_vector = measured.as_vector()
        result = planner_step(measured, commands, field, solver_state, cfg)
        trajectory = result.full_trajectory
        next_state = trajectory.state[:, 1]
        next_velocity = trajectory.derived_velocity[:, 0]
        x0_error = (trajectory.state[:, 0] - measured_vector).abs().amax(dim=-1)
        pending_vector = torch.cat(
            (
                result.pending_reference.root_pos_w,
                result.pending_reference.root_rpy_w,
                result.pending_reference.joint_angles,
            ),
            dim=-1,
        )
        x1_error = (trajectory.state[:, 1] - pending_vector).abs().amax(dim=-1)
        measured = JointMpcRtiState(
            root_pos_w=next_state[:, :3],
            root_rpy_w=next_state[:, 3:6],
            joint_pos=next_state[:, 6:],
            root_lin_vel_b=next_velocity[:, :3],
            root_ang_vel_b=next_velocity[:, 3:6],
            joint_vel=next_velocity[:, 6:],
        )
        solver_state = result.solver_state
        state_rows.append(next_state)
        root_rpy_rows.append(measured.root_rpy_w)
        joint_rows.append(measured.joint_pos)
        foot_rows.append(trajectory.foot_pos_w[:, 1])
        contact_rows.append(trajectory.contact_state[:, 1])
        phase_rows.append(torch.remainder(phase_rows[-1] + 1, 24))
        alpha_rows.append(trajectory.line_search_alpha)
        valid_rows.append(trajectory.valid)
        queried = query_world(field, trajectory.foot_pos_w[:, 1])
        map_valid_rows.append(queried.valid.all(dim=-1))
        x0_errors.append(x0_error)
        x1_errors.append(x1_error)
        cold_start_rows.append(trajectory.cold_start)
        warm_start_rows.append(trajectory.warm_start)
        timestamps.append(torch.full((batch,), (step_index + 1) * float(cfg.runtime.dt), device=device))

    try:
        from .joint_metrics import JointMetricTrace
    except ImportError:  # pragma: no cover - direct script execution
        from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import JointMetricTrace

    roots = torch.stack(state_rows, dim=1)
    feet = torch.stack(foot_rows, dim=1)
    contacts = torch.stack([fixed_trot_schedule(torch.zeros(batch, dtype=torch.long, device=device)).stance[:, 0]] + contact_rows, dim=1)
    anchors = feet.clone()
    for node in range(1, int(feet.shape[1])):
        continuing = contacts[:, node] & contacts[:, node - 1]
        anchors[:, node] = torch.where(continuing[..., None], anchors[:, node - 1], feet[:, node])
    root_rpy = torch.stack(root_rpy_rows, dim=1)
    joints = torch.stack(joint_rows, dim=1)
    phase = torch.stack(phase_rows, dim=1)
    valid = torch.stack(valid_rows, dim=1)
    map_valid = torch.stack(map_valid_rows, dim=1)
    field_height = torch.zeros_like(feet[..., 0])
    return JointMetricTrace(
        root_pos_w=roots[..., :3],
        root_rpy_w=root_rpy,
        joint_pos=joints,
        foot_pos_w=feet,
        contact_state=contacts,
        command_body=commands[:, None].expand(-1, roots.shape[1], -1),
        gait_phase=phase,
        foot_height_w=field_height,
        foot_small_distance_m=torch.ones_like(field_height),
        part_collision={name: torch.zeros(batch, roots.shape[1], dtype=torch.bool, device=device) for name in ("foot", "knee", "calf", "thigh", "base")},
        line_alpha=torch.stack(alpha_rows, dim=1),
        nominal_root_pos_w=roots[..., :3],
        nominal_root_rpy_w=root_rpy,
        valid=valid,
        map_valid=map_valid,
        timestamps=torch.stack(timestamps, dim=1),
        dt=float(cfg.runtime.dt),
        stance_anchor_w=anchors,
        x0_injection_error=torch.stack(x0_errors, dim=1),
        published_x1_error=torch.stack(x1_errors, dim=1),
        cold_start=torch.stack(cold_start_rows, dim=1),
        warm_start=torch.stack(warm_start_rows, dim=1),
        warm_cache_invariant_fault=torch.zeros_like(valid),
    )


def _small_detector_row(field, geometry, contact: torch.Tensor, previous_contact: torch.Tensor):
    from extension.joint_mpc_rti.terrain.query import query_world

    points = {
        "foot": (geometry.foot_pos_w, 0.022),
        "knee": (geometry.knee_pos_w, 0.025),
        "calf": (geometry.shank_samples_w, 0.025),
        "thigh": (geometry.thigh_samples_w, 0.025),
        "base": (geometry.body_samples_w, 0.04),
    }
    collision: dict[str, torch.Tensor] = {}
    penetration: dict[str, torch.Tensor] = {}
    foot_query = None
    for name, (part_points, radius) in points.items():
        batch = int(part_points.shape[0])
        flat = part_points.reshape(batch, -1, 3)
        query = query_world(field, flat)
        inside_small = query.small_distance_m <= 0.0
        depth = torch.where(
            inside_small,
            (query.height_w + float(radius) - flat[..., 2]).clamp_min(0.0),
            torch.zeros_like(query.height_w),
        )
        collision[name] = (depth > 0.0).any(dim=1)
        penetration[name] = depth.amax(dim=1)
        if name == "foot":
            foot_query = query
    assert foot_query is not None
    foot_small = (foot_query.small_distance_m <= 0.0).reshape(contact.shape)
    touchdown = contact & ~previous_contact
    surface_gap = geometry.foot_pos_w[..., 2] - foot_query.height_w.reshape(contact.shape) - 0.022
    return {
        "foot_height": foot_query.height_w.reshape(contact.shape),
        "foot_distance": foot_query.small_distance_m.reshape(contact.shape),
        "collision": collision,
        "penetration": penetration,
        "touchdown_on_small": (touchdown & foot_small).any(dim=1),
        "stance_on_small": (contact & foot_small).any(dim=1),
        "airborne_touchdown": (touchdown & (surface_gap.abs() > 0.012)).any(dim=1),
        "map_valid": foot_query.valid.reshape(contact.shape).all(dim=1),
    }


def simulate_small_trace(
    cells: tuple[AcceptanceCell, ...],
    *,
    steps: int,
    device: str = "cpu",
    cfg=None,
):
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule
    from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
    from extension.joint_mpc_rti.planner import step as planner_step
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.types import JointMpcRtiState
    try:
        from .helpers import make_state
        from .joint_metrics import JointMetricTrace
    except ImportError:  # pragma: no cover
        from Go2Pvcnn.tests.joint_mpc_rti.helpers import make_state
        from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import JointMetricTrace

    if not cells or any(cell.scenario != "small" for cell in cells):
        raise ValueError("simulate_small_trace requires non-empty small cells")
    phase = int(cells[0].phase or 0)
    if any(int(cell.phase or 0) != phase for cell in cells):
        raise ValueError("simulate_small_trace requires one shared obstacle-entry phase")
    commands = torch.tensor([cell.command for cell in cells], dtype=torch.float32, device=device)
    offsets = torch.tensor([float(cell.offset or 0.0) for cell in cells], dtype=torch.float32, device=device)
    shapes = tuple(str(cell.shape) for cell in cells)
    cfg = JointMpcRtiCfg() if cfg is None else cfg
    batch = len(cells)
    measured = make_state(batch, device=device)
    solver_state = None
    flat_field = build_field_batch(
        height_w=torch.zeros(batch, 151, 151, device=device),
        semantic_id=torch.zeros(batch, 151, 151, dtype=torch.long, device=device),
        origin_w=torch.zeros(batch, 3, device=device),
        yaw_w=torch.zeros(batch, device=device),
        timestamp=torch.zeros(batch, device=device),
        version=torch.ones(batch, dtype=torch.long, device=device),
        resolution=0.05,
        small_ids=(1,),
        large_ids=(2,),
        terrain_cfg=cfg.terrain,
    )
    initial_geometry = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos)
    initial_phase = torch.zeros(batch, dtype=torch.long, device=device)
    initial_contact = fixed_trot_schedule(initial_phase).stance[:, 0]
    initial_detector = _small_detector_row(flat_field, initial_geometry, initial_contact, initial_contact)
    states = [measured.as_vector()]
    feet = [initial_geometry.foot_pos_w]
    contacts = [initial_contact]
    phase_rows = [initial_phase]
    alphas = [torch.ones(batch, device=device)]
    valids = [torch.ones(batch, dtype=torch.bool, device=device)]
    timestamps = [torch.zeros(batch, device=device)]
    foot_heights = [initial_detector["foot_height"]]
    foot_distances = [initial_detector["foot_distance"]]
    map_valids = [initial_detector["map_valid"]]
    collision_rows = {name: [initial_detector["collision"][name]] for name in initial_detector["collision"]}
    penetration_rows = {name: [initial_detector["penetration"][name]] for name in initial_detector["penetration"]}
    touchdown_rows = [initial_detector["touchdown_on_small"]]
    stance_rows = [initial_detector["stance_on_small"]]
    airborne_rows = [initial_detector["airborne_touchdown"]]
    x0_errors = [torch.zeros(batch, device=device)]
    x1_errors = [torch.zeros(batch, device=device)]
    cold_rows = [torch.zeros(batch, dtype=torch.bool, device=device)]
    warm_rows = [torch.zeros(batch, dtype=torch.bool, device=device)]

    def advance(field, step_index: int) -> None:
        nonlocal measured, solver_state
        measured_vector = measured.as_vector()
        result = planner_step(measured, commands, field, solver_state, cfg)
        trajectory = result.full_trajectory
        next_state = trajectory.state[:, 1]
        next_velocity = trajectory.derived_velocity[:, 0]
        pending = torch.cat(
            (result.pending_reference.root_pos_w, result.pending_reference.root_rpy_w, result.pending_reference.joint_angles),
            dim=-1,
        )
        measured = JointMpcRtiState(
            root_pos_w=next_state[:, :3],
            root_rpy_w=next_state[:, 3:6],
            joint_pos=next_state[:, 6:],
            root_lin_vel_b=next_velocity[:, :3],
            root_ang_vel_b=next_velocity[:, 3:6],
            joint_vel=next_velocity[:, 6:],
        )
        solver_state = result.solver_state
        geometry = go2_fk(measured.root_pos_w, measured.root_rpy_w, measured.joint_pos)
        contact = trajectory.contact_state[:, 1]
        detector = _small_detector_row(field, geometry, contact, contacts[-1])
        states.append(next_state)
        feet.append(geometry.foot_pos_w)
        contacts.append(contact)
        phase_rows.append(solver_state.gait_phase.clone())
        alphas.append(trajectory.line_search_alpha)
        valids.append(trajectory.valid)
        timestamps.append(torch.full((batch,), (step_index + 1) * float(cfg.runtime.dt), device=device))
        foot_heights.append(detector["foot_height"])
        foot_distances.append(detector["foot_distance"])
        map_valids.append(detector["map_valid"])
        for name in collision_rows:
            collision_rows[name].append(detector["collision"][name])
            penetration_rows[name].append(detector["penetration"][name])
        touchdown_rows.append(detector["touchdown_on_small"])
        stance_rows.append(detector["stance_on_small"])
        airborne_rows.append(detector["airborne_touchdown"])
        x0_errors.append((trajectory.state[:, 0] - measured_vector).abs().amax(dim=-1))
        x1_errors.append((trajectory.state[:, 1] - pending).abs().amax(dim=-1))
        cold_rows.append(trajectory.cold_start)
        warm_rows.append(trajectory.warm_start)

    for step_index in range(phase):
        advance(flat_field, step_index)

    field, obstacle_center, obstacle_radius = build_small_obstacle_field(
        commands=commands,
        shapes=shapes,
        offsets=offsets,
        origin_xy_w=measured.root_pos_w[:, :2],
        device=device,
        terrain_cfg=cfg.terrain,
    )
    for obstacle_step in range(int(steps)):
        field, _, _ = build_small_obstacle_field(
            commands=commands,
            shapes=shapes,
            offsets=offsets,
            origin_xy_w=measured.root_pos_w[:, :2],
            obstacle_center_xy_w=obstacle_center,
            device=device,
            terrain_cfg=cfg.terrain,
        )
        advance(field, phase + obstacle_step)

    state = torch.stack(states, dim=1)
    foot = torch.stack(feet, dim=1)
    contact = torch.stack(contacts, dim=1)
    anchors = foot.clone()
    for node in range(1, int(foot.shape[1])):
        continuing = contact[:, node] & contact[:, node - 1]
        anchors[:, node] = torch.where(continuing[..., None], anchors[:, node - 1], foot[:, node])
    crossing = strict_crossing_event(
        state[..., :2], commands[:, :2], obstacle_center, radius_m=obstacle_radius
    ).success.to(state.dtype)
    return JointMetricTrace(
        root_pos_w=state[..., :3],
        root_rpy_w=state[..., 3:6],
        joint_pos=state[..., 6:],
        foot_pos_w=foot,
        contact_state=contact,
        command_body=commands[:, None].expand(-1, state.shape[1], -1),
        gait_phase=torch.stack(phase_rows, dim=1),
        foot_height_w=torch.stack(foot_heights, dim=1),
        foot_small_distance_m=torch.stack(foot_distances, dim=1),
        part_collision={name: torch.stack(rows, dim=1) for name, rows in collision_rows.items()},
        line_alpha=torch.stack(alphas, dim=1),
        nominal_root_pos_w=state[..., :3],
        nominal_root_rpy_w=state[..., 3:6],
        valid=torch.stack(valids, dim=1),
        map_valid=torch.stack(map_valids, dim=1),
        timestamps=torch.stack(timestamps, dim=1),
        dt=float(cfg.runtime.dt),
        stance_anchor_w=anchors,
        strict_cross_success=crossing,
        touchdown_on_small=torch.stack(touchdown_rows, dim=1),
        stance_on_small=torch.stack(stance_rows, dim=1),
        airborne_touchdown=torch.stack(airborne_rows, dim=1),
        part_penetration_m={name: torch.stack(rows, dim=1) for name, rows in penetration_rows.items()},
        x0_injection_error=torch.stack(x0_errors, dim=1),
        published_x1_error=torch.stack(x1_errors, dim=1),
        cold_start=torch.stack(cold_rows, dim=1),
        warm_start=torch.stack(warm_rows, dim=1),
        warm_cache_invariant_fault=torch.zeros_like(torch.stack(valids, dim=1)),
    )


def run_flat_acceptance(
    *,
    cells: tuple[AcceptanceCell, ...],
    steps: int,
    device: str = "cpu",
    cell_batch_size: int | None = None,
) -> AcceptanceReport:
    try:
        from .joint_metrics import evaluate_trace
    except ImportError:  # pragma: no cover - direct script execution
        from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import evaluate_trace

    if not cells or any(cell.scenario != "flat" for cell in cells):
        raise ValueError("run_flat_acceptance requires non-empty flat cells")
    chunk_size = len(cells) if cell_batch_size is None else int(cell_batch_size)
    if chunk_size <= 0:
        raise ValueError("cell_batch_size must be positive")
    reports = []
    for start in range(0, len(cells), chunk_size):
        chunk = cells[start : start + chunk_size]
        commands = torch.tensor([cell.command for cell in chunk], dtype=torch.float32, device=device)
        trace = simulate_flat_trace(commands, steps=steps, device=device)
        for local_index, cell in enumerate(chunk):
            metric_report = evaluate_trace(
                _slice_trace(trace, local_index), scenario="flat", key=cell.key
            )
            reports.append(
                CellReport(cell=cell, metrics=metric_report.metrics, passed=metric_report.passed)
            )
            emit_cell_progress(
                cell=cell,
                index=start + local_index + 1,
                total=len(cells),
                passed=metric_report.passed,
            )
    try:
        code_ref = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_ref = "unknown"
    return AcceptanceReport(stage="flat", code_ref=code_ref, cells=tuple(reports))


def run_small_acceptance(
    *,
    cells: tuple[AcceptanceCell, ...],
    steps: int,
    device: str = "cpu",
    cell_batch_size: int | None = None,
) -> AcceptanceReport:
    try:
        from .joint_metrics import evaluate_trace
    except ImportError:  # pragma: no cover
        from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import evaluate_trace

    if not cells or any(cell.scenario != "small" for cell in cells):
        raise ValueError("run_small_acceptance requires non-empty small cells")
    chunk_size = len(cells) if cell_batch_size is None else int(cell_batch_size)
    if chunk_size <= 0:
        raise ValueError("cell_batch_size must be positive")
    reports: list[CellReport | None] = [None] * len(cells)
    for start in range(0, len(cells), chunk_size):
        chunk = cells[start : start + chunk_size]
        phase_groups: dict[int, list[tuple[int, AcceptanceCell]]] = {}
        for local_index, cell in enumerate(chunk):
            phase_groups.setdefault(int(cell.phase or 0), []).append((local_index, cell))
        for indexed_group in phase_groups.values():
            group = tuple(cell for _, cell in indexed_group)
            trace = simulate_small_trace(group, steps=steps, device=device)
            for group_index, (local_index, cell) in enumerate(indexed_group):
                metric_report = evaluate_trace(
                    _slice_trace(trace, group_index), scenario="small", key=cell.key
                )
                reports[start + local_index] = CellReport(
                    cell=cell,
                    metrics=metric_report.metrics,
                    passed=metric_report.passed,
                )
                emit_cell_progress(
                    cell=cell,
                    index=start + local_index + 1,
                    total=len(cells),
                    passed=metric_report.passed,
                )
    try:
        code_ref = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_ref = "unknown"
    if any(report is None for report in reports):
        raise RuntimeError("small acceptance did not produce every requested cell")
    return AcceptanceReport(
        stage="small",
        code_ref=code_ref,
        cells=tuple(report for report in reports if report is not None),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "flat-small":
        raise SystemExit("flat-small execution is added in Task 15")
    all_cells = build_cells(
        stage=args.stage,
        all_commands=bool(args.all_commands or args.formal),
        all_shapes=bool(args.all_shapes or args.formal),
        all_phases=bool(args.all_phases or args.formal),
        all_offsets=bool(args.all_offsets or args.formal),
        ranked_cells=args.ranked_cells,
    )
    formal = bool(args.all_commands or args.formal)
    shard_metadata = None
    merged_shards = None
    if args.merge_shard_reports:
        if args.shard_count is not None or args.shard_index is not None:
            raise SystemExit("merge mode cannot also select a shard")
        shard_payloads = tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.merge_shard_reports
        )
        report = merge_acceptance_shard_payloads(
            shard_payloads,
            expected_cells=all_cells,
            stage=args.stage,
        )
        merged_shards = len(shard_payloads)
    else:
        if (args.shard_count is None) != (args.shard_index is None):
            raise SystemExit("--shard-count and --shard-index must be provided together")
        shard_count = 1 if args.shard_count is None else int(args.shard_count)
        shard_index = 0 if args.shard_index is None else int(args.shard_index)
        cells = select_cell_shard(
            all_cells,
            shard_count=shard_count,
            shard_index=shard_index,
        )
        if not cells:
            raise SystemExit("selected shard contains no cells")
        shard_metadata = ShardMetadata(
            index=shard_index,
            count=shard_count,
            total_cells=len(all_cells),
            selected_cells=len(cells),
        )
        cell_batch_size = args.cell_batch_size
        if cell_batch_size is None and formal:
            cell_batch_size = 40
        runner = run_flat_acceptance if args.stage == "flat" else run_small_acceptance
        report = runner(
            cells=cells,
            steps=int(args.steps),
            device=str(args.device),
            cell_batch_size=cell_batch_size,
        )
    gate = require_flat_gate(report) if args.stage == "flat" else require_small_gate(report)
    payload = {
        **report.to_dict(),
        "gate": asdict(gate),
        "formal_complete": bool(args.merge_shard_reports) or shard_metadata.count == 1,
    }
    if shard_metadata is not None:
        payload["shard"] = asdict(shard_metadata)
    if merged_shards is not None:
        payload["merged_shards"] = merged_shards
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if gate.passed else 2


def metric_registry(scenario: str) -> frozenset[str]:
    return applicable_metrics(scenario)


def build_cells(
    *,
    stage: str,
    all_commands: bool = False,
    all_shapes: bool = False,
    all_phases: bool = False,
    all_offsets: bool = False,
    ranked_cells: int | None = None,
) -> tuple[AcceptanceCell, ...]:
    if stage not in ("flat", "small", "flat-small"):
        raise ValueError("stage must be flat, small, or flat-small")
    commands = COMMANDS if all_commands else RANKED_COMMANDS
    scenarios = ("flat", "small") if stage == "flat-small" else (stage,)
    shapes = SMALL_SHAPES if all_shapes else (SMALL_SHAPES[0],)
    phases = SMALL_PHASES if all_phases else (SMALL_PHASES[0],)
    offsets = SMALL_OFFSETS if all_offsets else (0.0,)
    cells: list[AcceptanceCell] = []
    for scenario in scenarios:
        for command in commands:
            if scenario == "flat":
                cells.append(AcceptanceCell(scenario=scenario, command=command))
            else:
                cells.extend(
                    AcceptanceCell(
                        scenario=scenario,
                        command=command,
                        phase=phase,
                        shape=shape,
                        offset=offset,
                    )
                    for shape in shapes
                    for phase in phases
                    for offset in offsets
                )
    if ranked_cells is not None:
        if ranked_cells <= 0:
            raise ValueError("ranked_cells must be positive")
        cells = cells[:ranked_cells]
    return tuple(cells)


def emit_cell_progress(*, cell: AcceptanceCell, index: int, total: int, passed: bool) -> None:
    print(
        json.dumps(
            {
                "event": "cell_complete",
                "index": int(index),
                "total": int(total),
                "key": cell.key,
                "passed": bool(passed),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified joint MPC behavior acceptance.")
    parser.add_argument("--stage", choices=("flat", "small", "flat-small"), required=True)
    parser.add_argument("--ranked-cells", type=int)
    parser.add_argument("--all-commands", action="store_true")
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--all-phases", action="store_true")
    parser.add_argument("--all-offsets", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--steps", type=int, default=144)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--cell-batch-size", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--merge-shard-reports", type=Path, nargs="+")
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


__all__ = [
    "AcceptanceCell",
    "AcceptanceReport",
    "CellReport",
    "ShardMetadata",
    "acceptance_report_from_dict",
    "build_cells",
    "build_small_obstacle_field",
    "emit_cell_progress",
    "GateResult",
    "metric_registry",
    "merge_acceptance_shard_payloads",
    "parse_args",
    "require_flat_gate",
    "require_small_gate",
    "require_gate",
    "run_flat_acceptance",
    "run_small_acceptance",
    "select_cell_shard",
    "simulate_flat_trace",
    "simulate_small_trace",
    "strict_crossing_event",
]


if __name__ == "__main__":
    raise SystemExit(main())
