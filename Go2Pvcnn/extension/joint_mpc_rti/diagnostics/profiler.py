"""Preallocated refresh-stage timing without planner-side synchronization."""

from __future__ import annotations

from collections.abc import Callable
import time

import torch

from .metrics import timing_summary


STAGE_NAMES = (
    "field",
    "selector",
    "region",
    "nominal_ik",
    "linearization",
    "scan_qp",
    "line_search_safety",
    "cache_diagnostics",
)


class RefreshStageProfiler:
    """Record one refresh using fixed events; reduction happens after the call."""

    def __init__(self, *, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._next = 0
        if self.device.type == "cuda":
            self._start = torch.cuda.Event(enable_timing=True)
            self._ends = tuple(
                torch.cuda.Event(enable_timing=True) for _ in STAGE_NAMES
            )
            self._start.record()
            self._host_last = 0.0
            self._host_ms: list[float] = []
            self._region_events = tuple(
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(2)
            )
        else:
            self._start = None
            self._ends = ()
            self._host_last = time.perf_counter()
            self._host_ms = []
            self._region_events = ()
        self._region_index = 0
        self._region_host_start = 0.0
        self._region_host_ms: list[float] = []

    def begin_region(self) -> None:
        if self._region_index >= 2:
            raise ValueError("at most two region segments are supported")
        if self.device.type == "cuda":
            self._region_events[self._region_index][0].record()
        else:
            self._region_host_start = time.perf_counter()

    def end_region(self) -> None:
        if self.device.type == "cuda":
            self._region_events[self._region_index][1].record()
        else:
            self._region_host_ms.append(
                (time.perf_counter() - self._region_host_start) * 1000.0
            )
        self._region_index += 1

    def record(self, stage: str) -> None:
        if self._next >= len(STAGE_NAMES):
            raise ValueError("all refresh stages are already recorded")
        expected = STAGE_NAMES[self._next]
        if stage != expected:
            raise ValueError(f"expected stage {expected!r}, got {stage!r}")
        if self.device.type == "cuda":
            self._ends[self._next].record()
        else:
            now = time.perf_counter()
            self._host_ms.append((now - self._host_last) * 1000.0)
            self._host_last = now
        self._next += 1

    def elapsed_ms(self) -> dict[str, float]:
        if self._next != len(STAGE_NAMES):
            raise RuntimeError("refresh stage recording is incomplete")
        if self.device.type == "cuda":
            previous = self._start
            values: list[float] = []
            for event in self._ends:
                values.append(float(previous.elapsed_time(event)))
                previous = event
            region_ms = sum(
                float(start.elapsed_time(end))
                for start, end in self._region_events[: self._region_index]
            )
        else:
            values = list(self._host_ms)
            region_ms = sum(self._region_host_ms)
        values[1] = max(values[1] - region_ms, 0.0)
        values[2] = region_ms
        return dict(zip(STAGE_NAMES, values))


def benchmark_cuda_replay(
    replay: Callable[[], None], *, steps: int, warmup: int
) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the joint MPC performance probe")
    for _ in range(int(warmup)):
        replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(int(steps))]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(int(steps))]
    for start, end in zip(starts, ends):
        start.record()
        replay()
        end.record()
    torch.cuda.synchronize()
    return timing_summary(
        [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    )


__all__ = ["RefreshStageProfiler", "STAGE_NAMES", "benchmark_cuda_replay"]
