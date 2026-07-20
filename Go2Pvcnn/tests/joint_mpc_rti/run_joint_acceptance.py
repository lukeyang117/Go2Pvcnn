"""Unified flat/small acceptance entrypoint and report schema."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

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
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args(argv)


__all__ = [
    "AcceptanceCell",
    "AcceptanceReport",
    "CellReport",
    "build_cells",
    "emit_cell_progress",
    "metric_registry",
    "parse_args",
]
