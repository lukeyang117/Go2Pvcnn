"""Unified flat/small acceptance entrypoint and report schema."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

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
    (1.0, 0.5, 1.0),
    (-1.0, -0.5, -1.0),
)


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
class GateResult:
    stage: str
    passed: bool
    failures: tuple[str, ...]


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
    )


def run_flat_acceptance(*, cells: tuple[AcceptanceCell, ...], steps: int, device: str = "cpu") -> AcceptanceReport:
    try:
        from .joint_metrics import evaluate_trace
    except ImportError:  # pragma: no cover - direct script execution
        from Go2Pvcnn.tests.joint_mpc_rti.joint_metrics import evaluate_trace

    if not cells or any(cell.scenario != "flat" for cell in cells):
        raise ValueError("run_flat_acceptance requires non-empty flat cells")
    commands = torch.tensor([cell.command for cell in cells], dtype=torch.float32, device=device)
    trace = simulate_flat_trace(commands, steps=steps, device=device)
    reports = []
    for index, cell in enumerate(cells):
        metric_report = evaluate_trace(_slice_trace(trace, index), scenario="flat", key=cell.key)
        reports.append(CellReport(cell=cell, metrics=metric_report.metrics, passed=metric_report.passed))
        emit_cell_progress(cell=cell, index=index + 1, total=len(cells), passed=metric_report.passed)
    try:
        code_ref = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        code_ref = "unknown"
    return AcceptanceReport(stage="flat", code_ref=code_ref, cells=tuple(reports))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage != "flat":
        raise SystemExit("small and flat-small execution is added with the Task 14 terrain adapter")
    cells = build_cells(
        stage="flat",
        all_commands=bool(args.all_commands or args.formal),
        ranked_cells=args.ranked_cells,
    )
    report = run_flat_acceptance(cells=cells, steps=int(args.steps), device=str(args.device))
    gate = require_flat_gate(report)
    payload = {**report.to_dict(), "gate": asdict(gate)}
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
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


__all__ = [
    "AcceptanceCell",
    "AcceptanceReport",
    "CellReport",
    "build_cells",
    "emit_cell_progress",
    "GateResult",
    "metric_registry",
    "parse_args",
    "require_flat_gate",
    "require_gate",
    "run_flat_acceptance",
    "simulate_flat_trace",
]


if __name__ == "__main__":
    raise SystemExit(main())
