"""Deterministic command and timing protocols for the paper benchmark."""

from __future__ import annotations

import random


MIXED_VX = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
MIXED_VY = (-0.5, -0.25, 0.25, 0.5)
MIXED_YAW = (-1.0, -0.5, 0.0, 0.5, 1.0)
RUNWAY_VX = (0.25, 0.5, 1.0)


def mixed_velocity_commands() -> tuple[tuple[float, float, float], ...]:
    """Return the fixed 120-command mixed-velocity grid."""

    return tuple((vx, vy, yaw) for vx in MIXED_VX for vy in MIXED_VY for yaw in MIXED_YAW)


def runway_velocity_commands(layout_count: int, seed: int) -> tuple[tuple[float, float, float], ...]:
    """Assign the three runway speeds evenly and shuffle them reproducibly."""

    count = int(layout_count)
    if count <= 0:
        raise ValueError("layout_count must be positive")
    values = [RUNWAY_VX[index % len(RUNWAY_VX)] for index in range(count)]
    # A round uses each speed once per layout group.  For a partial group the
    # cyclic assignment keeps the imbalance at at most one sample.
    random.Random(int(seed)).shuffle(values)
    return tuple((float(vx), 0.0, 0.0) for vx in values)


def replan_boundary(step: int, interval: int = 23) -> bool:
    """Return whether ``step`` is reset or a completed 23-transition boundary."""

    value = int(step)
    cadence = int(interval)
    if value < 0:
        raise ValueError("step must be non-negative")
    if cadence <= 0:
        raise ValueError("interval must be positive")
    return value == 0 or value % cadence == 0
