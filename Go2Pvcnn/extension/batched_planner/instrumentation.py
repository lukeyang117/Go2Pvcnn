"""Lightweight planner-owned instrumentation helpers.

This module is intentionally small and dependency-free so it can be used from
both training and offline benchmark paths later on.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class StageTiming:
    """Aggregated timing for a named stage."""

    count: int
    total_s: float
    last_s: float
    min_s: float
    max_s: float

    @property
    def avg_s(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total_s / float(self.count)


@dataclass(frozen=True)
class PlannerTimingSummary:
    """Snapshot of planner timings at a point in time."""

    stages: dict[str, StageTiming]

    def format_compact(self, *, max_stages: int = 8) -> str:
        if not self.stages:
            return "PlannerTiming[]"

        # Sort by total time descending to surface hotspots.
        items = sorted(self.stages.items(), key=lambda kv: kv[1].total_s, reverse=True)
        parts: list[str] = []
        for name, st in items[: max(1, int(max_stages))]:
            parts.append(f"{name}={st.total_s * 1e3:.2f}ms/{st.count}")
        if len(items) > max_stages:
            parts.append(f"+{len(items) - max_stages} more")
        return "PlannerTiming[" + " ".join(parts) + "]"


class _StageAccumulator:
    __slots__ = ("count", "total_s", "last_s", "min_s", "max_s")

    def __init__(self) -> None:
        self.count = 0
        self.total_s = 0.0
        self.last_s = 0.0
        self.min_s = float("inf")
        self.max_s = 0.0

    def record(self, dt_s: float) -> None:
        dt_s = float(dt_s)
        self.count += 1
        self.total_s += dt_s
        self.last_s = dt_s
        if dt_s < self.min_s:
            self.min_s = dt_s
        if dt_s > self.max_s:
            self.max_s = dt_s

    def snapshot(self) -> StageTiming:
        if self.count <= 0:
            return StageTiming(count=0, total_s=0.0, last_s=0.0, min_s=0.0, max_s=0.0)
        return StageTiming(
            count=self.count,
            total_s=self.total_s,
            last_s=self.last_s,
            min_s=self.min_s if self.min_s != float("inf") else 0.0,
            max_s=self.max_s,
        )

    def reset(self) -> None:
        self.count = 0
        self.total_s = 0.0
        self.last_s = 0.0
        self.min_s = float("inf")
        self.max_s = 0.0


class _StageContext(AbstractContextManager):
    __slots__ = ("_instr", "_name", "_t0")

    def __init__(self, instr: "PlannerInstrumentation", name: str) -> None:
        self._instr = instr
        self._name = name
        self._t0: float | None = None

    def __enter__(self):
        self._t0 = self._instr._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        t0 = self._t0
        if t0 is None:
            return False
        dt = self._instr._clock() - t0
        self._instr._record(self._name, dt)
        return False


class _NoopStageContext(AbstractContextManager):
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class PlannerInstrumentation:
    """Collect lightweight stage-level timing for planner calls."""

    __slots__ = ("_enabled", "_clock", "_total", "_window")

    def __init__(self, *, enabled: bool = True, clock: Callable[[], float] | None = None) -> None:
        self._enabled = bool(enabled)
        self._clock = clock or perf_counter
        self._total: dict[str, _StageAccumulator] = {}
        self._window: dict[str, _StageAccumulator] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stage(self, name: str) -> AbstractContextManager:
        if not self._enabled:
            return _NoopStageContext()
        return _StageContext(self, str(name))

    def _record(self, name: str, dt_s: float) -> None:
        acc = self._total.get(name)
        if acc is None:
            acc = _StageAccumulator()
            self._total[name] = acc
        acc.record(dt_s)

        win = self._window.get(name)
        if win is None:
            win = _StageAccumulator()
            self._window[name] = win
        win.record(dt_s)

    def summary(self, *, window: bool = False, reset_window: bool = False) -> PlannerTimingSummary:
        src = self._window if window else self._total
        stages = {name: acc.snapshot() for name, acc in src.items() if acc.count > 0}
        if window and reset_window:
            for acc in self._window.values():
                acc.reset()
        return PlannerTimingSummary(stages=stages)


__all__ = ["PlannerInstrumentation", "PlannerTimingSummary", "StageTiming"]

