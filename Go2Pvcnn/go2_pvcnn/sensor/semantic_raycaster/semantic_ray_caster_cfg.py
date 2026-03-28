# Copyright (c) 2026, Go2Pvcnn contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`SemanticGridRayCaster`."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.utils import configclass


@configclass
class SemanticGridRayCasterCfg(RayCasterCfg):
    """Isaac Lab :class:`RayCasterCfg` plus per-mesh semantic ids for multi-mesh height scans.

    Extends the stock ray caster to allow multiple ``mesh_prim_paths`` (terrain + static obstacles).
    Each path must appear in :attr:`mesh_semantic_ids` with value 0 (terrain), 1 (small obstacle),
    or 2 (large obstacle). At init, all submeshes are merged into one warp mesh; each triangle stores
    that path's semantic id. At runtime a single ``raycast_mesh(..., return_face_id=True)`` resolves class.
    """

    mesh_semantic_ids: dict[str, int] = MISSING
    """Maps each entry of ``mesh_prim_paths`` to semantic id {0, 1, 2}."""

    height_scan_offset: float = 0.5
    """Subtracted from elevation per ray (matches :func:`isaaclab.envs.mdp.height_scan` default)."""

    def __post_init__(self):
        from go2_pvcnn.sensor.semantic_raycaster.semantic_ray_caster import SemanticGridRayCaster

        self.class_type = SemanticGridRayCaster
        for p in self.mesh_prim_paths:
            if p not in self.mesh_semantic_ids:
                raise ValueError(
                    f"mesh_semantic_ids must contain an entry for each mesh_prim_paths entry; missing: {p!r}"
                )
