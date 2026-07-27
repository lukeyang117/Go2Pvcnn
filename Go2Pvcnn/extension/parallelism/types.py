from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ParallelismState:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    foot_pos_w: Tensor | None = None


@dataclass(frozen=True)
class ParallelismTerrain:
    height_w: Tensor
    semantic_id: Tensor
    valid_mask: Tensor
    origin_w: Tensor
    yaw_w: Tensor
    resolution: float


@dataclass(frozen=True)
class TerrainQueryResult:
    height: Tensor
    semantic: Tensor
    valid: Tensor


@dataclass(frozen=True)
class ParallelismDiagnostics:
    candidate_center_w: Tensor
    candidate_w: Tensor
    candidate_score: Tensor
    candidate_valid: Tensor
    candidate_reject_bits: Tensor
    candidate_collision_bits: Tensor
    collision_ellipsoid_names: tuple[str, ...]
    collision_probe_count: int
    candidate_semantic: Tensor
    fk_touchdown_semantic: Tensor
    selected_index: Tensor


@dataclass(frozen=True)
class ParallelismTrajectory:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    valid: Tensor
    selected_foothold_w: Tensor
    selected_score: Tensor
    diagnostics: ParallelismDiagnostics


@dataclass(frozen=True)
class ParallelismReference:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    valid: Tensor
