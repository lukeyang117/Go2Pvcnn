"""CUDA-event timing helpers that synchronize only after the measured loop."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .metrics import timing_summary


def benchmark_cuda_replay(
    replay: Callable[[], None],
    *,
    steps: int,
    warmup: int,
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
    return timing_summary([float(start.elapsed_time(end)) for start, end in zip(starts, ends)])


__all__ = ["benchmark_cuda_replay"]
