"""Import-safe state adapter scaffolding for planner integration.

This module intentionally avoids Isaac Lab imports so it can be imported and
tested in a plain Python environment. The goal is to define the smallest
useful shape for future planner-to-Isaac state wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast


def _coerce_float_tuple(values: Sequence[float], expected_length: int, *, field_name: str) -> tuple[float, ...]:
    """Convert a sequence into a fixed-length tuple of floats."""
    coerced = tuple(float(value) for value in values)
    if len(coerced) != expected_length:
        raise ValueError(f"{field_name} must contain exactly {expected_length} values, got {len(coerced)}")
    return coerced


def normalize_vector3(values: Sequence[float]) -> tuple[float, float, float]:
    """Return a 3D vector as a float tuple."""
    return cast(tuple[float, float, float], _coerce_float_tuple(values, 3, field_name="vector3"))


def normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return a quaternion as a float tuple in w-last or w-first neutral form."""
    return cast(
        tuple[float, float, float, float],
        _coerce_float_tuple(values, 4, field_name="quaternion"),
    )


@dataclass(frozen=True, slots=True)
class IsaacStateAdapterConfig:
    """Configuration scaffold for future Isaac state extraction."""

    root_frame: str = "world"
    foot_frame_names: tuple[str, ...] = ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT")
    joint_name_prefix: str = ""


@dataclass(frozen=True, slots=True)
class IsaacStateSnapshot:
    """Minimal planner-facing robot state container."""

    root_position_w: tuple[float, float, float] | None = None
    root_quaternion_w: tuple[float, float, float, float] | None = None
    joint_positions: tuple[float, ...] = ()
    foot_positions_w: tuple[tuple[float, float, float], ...] = ()
    contact_state: tuple[bool, ...] = ()

    def is_ready(self) -> bool:
        """Return True when the snapshot contains the root pose."""
        return self.root_position_w is not None and self.root_quaternion_w is not None

    def missing_fields(self) -> tuple[str, ...]:
        """Return a tuple of missing required fields."""
        missing = []
        if self.root_position_w is None:
            missing.append("root_position_w")
        if self.root_quaternion_w is None:
            missing.append("root_quaternion_w")
        return tuple(missing)

