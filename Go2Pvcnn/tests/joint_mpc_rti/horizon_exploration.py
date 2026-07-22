from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .acceptance_thresholds import MetricCellResult


@dataclass(frozen=True)
class HorizonCandidate:
    horizon_steps: int
    half_cycle_steps: int


@dataclass(frozen=True)
class HorizonReport:
    candidate: HorizonCandidate
    metric_cells: tuple[MetricCellResult, ...]
    diagnostics: dict[str, float | int] = field(default_factory=dict)

    @property
    def all_applicable_metrics_pass(self) -> bool:
        return bool(self.metric_cells) and all(cell.passed for cell in self.metric_cells)


def make_horizon_candidates(horizons: tuple[int, ...]) -> tuple[HorizonCandidate, ...]:
    if any(horizon < 16 or horizon > 50 or horizon % 2 for horizon in horizons):
        raise ValueError("horizon candidates must be even and within [16, 50]")
    return tuple(
        HorizonCandidate(horizon_steps=horizon, half_cycle_steps=horizon // 2)
        for horizon in horizons
    )


def select_shortest_passing(reports: Sequence[HorizonReport]) -> HorizonReport:
    passing = [report for report in reports if report.all_applicable_metrics_pass]
    if not passing:
        raise RuntimeError("no horizon candidate passes the complete Stage A contract")
    return min(passing, key=lambda report: report.candidate.horizon_steps)


__all__ = [
    "HorizonCandidate",
    "HorizonReport",
    "make_horizon_candidates",
    "select_shortest_passing",
]
