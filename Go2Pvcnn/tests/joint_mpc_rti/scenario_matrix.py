"""Formal signed command and terrain-cell matrices for joint MPC acceptance."""

from __future__ import annotations

import itertools


VX = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0)
VY = (0.0, -0.3, 0.3, -0.5, 0.5)
YAW = (0.0, -0.5, 0.5, -1.0, 1.0)
COMMANDS = tuple(itertools.product(VX, VY, YAW))
SMALL_SHAPES = ("sphere", "cuboid", "cylinder", "capsule", "cone")
SMALL_PHASES = tuple(range(24))
SMALL_OFFSETS = (-0.24, -0.20, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24)


__all__ = ["COMMANDS", "SMALL_OFFSETS", "SMALL_PHASES", "SMALL_SHAPES", "VX", "VY", "YAW"]
