# Copyright (c) 2026, Go2Pvcnn contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Data container for :class:`SemanticGridRayCaster`."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData


@dataclass
class SemanticGridRayCasterData(RayCasterData):
    """Ray caster data plus elevation and semantic grids (Isaac ``height_scan``-style + class id)."""

    elevation_map: torch.Tensor | None = None
    """Per-ray height signal reshaped to grid, shape (N, H, W). Same construction as ``mdp.height_scan``."""

    semantic_map: torch.Tensor | None = None
    """Per-ray semantic class id reshaped to grid, shape (N, H, W). Values 0=terrain, 1=small, 2=big."""
