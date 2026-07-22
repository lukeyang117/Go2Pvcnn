"""Formal signed command and terrain-cell matrices for joint MPC acceptance."""

from __future__ import annotations

VX = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0)
VY = (0.0, -0.3, 0.3, -0.5, 0.5)
YAW = (0.0, -0.5, 0.5, -1.0, 1.0)
COMMANDS = (
    ((0.0, 0.0, 0.0),)
    + tuple((vx, 0.0, 0.0) for vx in VX if vx != 0.0)
    + tuple((0.0, vy, 0.0) for vy in VY if vy != 0.0)
    + tuple((0.0, 0.0, yaw) for yaw in YAW if yaw != 0.0)
)
SMALL_SHAPES = ("sphere", "cuboid", "cylinder", "capsule", "cone")
SMALL_PHASES = tuple(range(24))
SMALL_OFFSETS = (-0.24, -0.20, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24)


__all__ = ["COMMANDS", "SMALL_OFFSETS", "SMALL_PHASES", "SMALL_SHAPES", "VX", "VY", "YAW"]
