"""Small host-side metric reducers used outside the planner hot path."""

from __future__ import annotations

from collections.abc import Sequence


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * float(len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - float(lower)
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def timing_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    values = sorted(float(value) for value in samples_ms)
    total = sum(values)
    return {
        "total_ms": total,
        "mean_ms": total / max(len(values), 1),
        "p50_ms": _quantile(values, 0.50),
        "p95_ms": _quantile(values, 0.95),
        "p99_ms": _quantile(values, 0.99),
        "max_ms": values[-1] if values else 0.0,
    }


__all__ = ["timing_summary"]
