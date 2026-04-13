"""Import-safe marker adapter scaffolding for planner integration.

The dataclasses and helpers in this module describe the minimal marker payload
shape that future Isaac Lab visualization code can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast


def _normalize_rgba(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return a 4-channel RGBA tuple with values coerced to floats."""
    rgba = tuple(float(value) for value in values)
    if len(rgba) != 4:
        raise ValueError(f"rgba must contain exactly 4 values, got {len(rgba)}")
    return cast(tuple[float, float, float, float], rgba)


@dataclass(frozen=True, slots=True)
class IsaacMarkerAdapterConfig:
    """Configuration scaffold for future marker visualization."""

    namespace: str = "planner"
    scale: float = 0.05
    color_rgba: tuple[float, float, float, float] = (1.0, 0.4, 0.1, 1.0)

    def normalized_color(self) -> tuple[float, float, float, float]:
        """Return the configured color as a validated RGBA tuple."""
        return _normalize_rgba(self.color_rgba)


@dataclass(frozen=True, slots=True)
class MarkerSpec:
    """Minimal marker payload for a single visualization target."""

    name: str
    position_w: tuple[float, float, float]
    color_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    scale: float = 0.05

    def payload(self) -> dict[str, object]:
        """Return a plain dictionary payload suitable for downstream adapters."""
        return {
            "name": self.name,
            "position_w": self.position_w,
            "color_rgba": _normalize_rgba(self.color_rgba),
            "scale": float(self.scale),
        }


def marker_names(specs: Sequence[MarkerSpec]) -> tuple[str, ...]:
    """Return the marker names in input order."""
    return tuple(spec.name for spec in specs)


def marker_payloads(specs: Sequence[MarkerSpec]) -> tuple[dict[str, object], ...]:
    """Return plain dictionary payloads for the provided marker specs."""
    return tuple(spec.payload() for spec in specs)
