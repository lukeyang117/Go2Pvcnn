from __future__ import annotations

from itertools import product


STAGE_A_VX = (0.0, -0.2, 0.2, -0.4, 0.4, -0.6, 0.6, -0.8, 0.8, -1.0, 1.0)
STAGE_A_VY = (0.0, -0.3, 0.3, -0.5, 0.5)
STAGE_A_YAW = (0.0, -0.5, 0.5, -1.0, 1.0)


def stage_a_commands() -> tuple[tuple[float, float, float], ...]:
    return tuple(product(STAGE_A_VX, STAGE_A_VY, STAGE_A_YAW))


__all__ = ["STAGE_A_VX", "STAGE_A_VY", "STAGE_A_YAW", "stage_a_commands"]
