"""Single-contract Stage A metric runner.

The heavy probes remain responsible for producing raw traces.  This module is
the shared boundary that keys every result by the original command tuple and
refuses to treat omitted command cells as passing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

from .acceptance_thresholds import MetricCellResult, evaluate_metric_cell
from .joint_metrics import JointMetricTrace, accumulate_joint_metrics
from .scenario_matrix import STAGE_A_VX, STAGE_A_VY, STAGE_A_YAW, stage_a_commands


Command = tuple[float, float, float]


@dataclass(frozen=True)
class MetricCell:
    key: tuple[object, ...]
    command: Command
    values: dict[str, float | int | None]
    numerators: dict[str, int | float]
    denominators: dict[str, int | float]
    valid_count: int
    na_reason: str | None

    @property
    def result(self) -> MetricCellResult:
        return evaluate_metric_cell(tuple(str(value) for value in self.key), self.values)


@dataclass(frozen=True)
class StageAReport:
    cells: tuple[MetricCell, ...]

    @property
    def failures(self) -> tuple[MetricCellResult, ...]:
        return tuple(cell.result for cell in self.cells if not cell.result.passed)

    @property
    def passed(self) -> bool:
        return bool(self.cells) and all(cell.na_reason is None for cell in self.cells) and not self.failures


def build_stage_a_cells(scenario: str, *, shape: str | None = None) -> tuple[MetricCell, ...]:
    """Create the complete fixed command matrix before any trace is attached."""
    return tuple(
        MetricCell(
            key=(scenario, shape, command[0], command[1], command[2]),
            command=command,
            values={},
            numerators={},
            denominators={},
            valid_count=0,
            na_reason="trace not attached",
        )
        for command in stage_a_commands()
    )


def metric_cell_from_trace(
    *,
    key: tuple[object, ...],
    command: Command,
    trace: JointMetricTrace,
    extra_values: dict[str, float | int | None] | None = None,
    na_reason: str | None = None,
) -> MetricCell:
    values = accumulate_joint_metrics(trace)
    if extra_values:
        values.update(extra_values)
    valid = int(trace.valid.to(dtype=trace.root_pos_w.dtype).sum().item())
    denominator = int(trace.valid.numel())
    return MetricCell(
        key=key,
        command=command,
        values=values,
        numerators={name: value for name, value in values.items() if isinstance(value, (int, float))},
        denominators={name: denominator for name in values},
        valid_count=valid,
        na_reason=na_reason,
    )


def require_complete_stage_a_matrix(
    report: StageAReport,
    *,
    scenarios: Sequence[str],
    commands: Sequence[Command] = stage_a_commands(),
) -> None:
    expected = {(scenario, command) for scenario in scenarios for command in commands}
    observed = {(str(cell.key[0]), cell.command) for cell in report.cells}
    missing = sorted(expected - observed)
    assert not missing, f"missing applicable command cells: {missing[:5]}"


def run_stage_a(
    cells: Iterable[MetricCell],
    *,
    scenarios: Sequence[str],
    commands: Sequence[Command] = stage_a_commands(),
) -> StageAReport:
    """Unify attached probe cells and enforce the universal Stage A matrix gate."""
    report = StageAReport(cells=tuple(cells))
    require_complete_stage_a_matrix(report, scenarios=scenarios, commands=commands)
    assert all(cell.na_reason is None for cell in report.cells), "Stage A contains unattached metric cells"
    return report


def report_to_json(report: StageAReport) -> dict[str, object]:
    return {
        "passed": report.passed,
        "cell_count": len(report.cells),
        "failure_count": len(report.failures),
        "cells": [
            {
                "key": list(cell.key),
                "command": list(cell.command),
                "values": cell.values,
                "numerators": cell.numerators,
                "denominators": cell.denominators,
                "valid_count": cell.valid_count,
                "na_reason": cell.na_reason,
                "passed": cell.result.passed,
                "failures": list(cell.result.failures),
            }
            for cell in report.cells
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the shared Stage A matrix contract")
    parser.add_argument("--json", action="store_true", help="print the contract matrix as JSON")
    args = parser.parse_args()
    report = StageAReport(cells=build_stage_a_cells("unattached"))
    if args.json:
        print(json.dumps(report_to_json(report), sort_keys=True))
    else:
        print(f"stage_a_cells={len(report.cells)} attached=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Command",
    "MetricCell",
    "StageAReport",
    "build_stage_a_cells",
    "metric_cell_from_trace",
    "require_complete_stage_a_matrix",
    "report_to_json",
    "run_stage_a",
]
