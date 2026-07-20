"""Unified flat/small acceptance entrypoint."""

from __future__ import annotations

from .joint_metrics import applicable_metrics


def metric_registry(scenario: str) -> frozenset[str]:
    return applicable_metrics(scenario)


__all__ = ["metric_registry"]
