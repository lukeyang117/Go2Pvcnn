"""Native batched-together planner tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

T116_CANDIDATE_COUNT = 5
HARD_REASON_NAMES = (
    "boundary_invalid",
    "path_collision",
    "body_hard_collision",
    "leg_hard_collision",
    "crossing_not_grounded",
    "touchdown_on_small",
    "front_foot_small_collision",
    "rear_foot_small_collision",
    "per_leg_foot_small_collision",
    "base_small_penetration",
    "touchdown_on_large",
    "foot_large_collision",
    "direction_violation",
)
HARD_REASON_COUNT = len(HARD_REASON_NAMES)
HARD_REASON_BOUNDARY_INVALID = 0
HARD_REASON_PATH_COLLISION = 1
HARD_REASON_BODY_HARD_COLLISION = 2
HARD_REASON_LEG_HARD_COLLISION = 3
HARD_REASON_CROSSING_NOT_GROUNDED = 4
HARD_REASON_TOUCHDOWN_ON_SMALL = 5
HARD_REASON_FRONT_FOOT_SMALL_COLLISION = 6
HARD_REASON_REAR_FOOT_SMALL_COLLISION = 7
HARD_REASON_PER_LEG_FOOT_SMALL_COLLISION = 8
HARD_REASON_BASE_SMALL_PENETRATION = 9
HARD_REASON_TOUCHDOWN_ON_LARGE = 10
HARD_REASON_FOOT_LARGE_COLLISION = 11
HARD_REASON_DIRECTION_VIOLATION = 12

HIP_HEIGHT = 0.30
HIP_OFFSET_Y = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
TOGETHER_TERRAIN_REFERENCE_STENCIL_RADIUS_M = 0.12
TOGETHER_TERRAIN_REFERENCE_STENCIL_STEP_M = 0.04
HIP_OFFSETS_ARRAY = torch.tensor(
    (
        (0.1934, 0.0465, 0.0),
        (0.1934, -0.0465, 0.0),
        (-0.1934, 0.0465, 0.0),
        (-0.1934, -0.0465, 0.0),
    ),
    dtype=torch.float32,
)


def _validate_optional_shape(value: Tensor | None, shape: tuple[int, ...], name: str) -> None:
    if value is not None and (value.ndim != len(shape) or tuple(value.shape) != shape):
        raise ValueError(f"{name} must have shape {shape}")


class TogetherPlannerStatus(IntEnum):
    OK = 0
    ALL_INFEASIBLE = 1
    INVALID_CONFIG = 2


class TogetherTerrainSemanticId(IntEnum):
    TERRAIN = 0
    SMALL = 1
    LARGE = 2


@dataclass(frozen=True)
class TogetherRobotState:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    joint_angles: Tensor | None = None
    foot_vel: Tensor | None = None


@dataclass(frozen=True)
class TogetherContactSchedule:
    contact_state: Tensor
    touchdown_mask: Tensor
    touchdown_frames: Tensor
    horizon_steps: int
    dt: float
    event_cap: int = 2


@dataclass(frozen=True)
class TogetherPlannerResult:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    joint_angles: Tensor
    contact_state: Tensor
    touchdown_seq: Tensor
    touchdown_mask: Tensor
    cost_total: Tensor
    cost_breakdown: dict[str, Tensor]
    status: Tensor
    feasible: Tensor
    safe_fallback: Tensor
    joint_limit_violation: Tensor
    workspace_margin: Tensor
    support_xy: Tensor
    support_height: Tensor
    support_slope: Tensor
    state_mode: Tensor | None = None
    small_strategy_outcome: Tensor | None = None
    selected_route_offset: Tensor | None = None
    semantic_candidate_costs: Tensor | None = None
    mode: Tensor | None = None
    selected_beta: Tensor | None = None
    selected_route: Tensor | None = None
    direction_id: Tensor | None = None
    small_front_s: Tensor | None = None
    small_back_s: Tensor | None = None
    small_top_z: Tensor | None = None
    command_direction_violation: Tensor | None = None
    cross_small_success: Tensor | None = None
    body_min_clearance: Tensor | None = None
    leg_min_clearance: Tensor | None = None
    front_touchdown_ground_gap: Tensor | None = None
    rear_touchdown_ground_gap: Tensor | None = None
    touchdown_on_small_count: Tensor | None = None
    front_foot_small_collision_count: Tensor | None = None
    rear_foot_small_collision_count: Tensor | None = None
    front_foot_min_clearance_to_small: Tensor | None = None
    rear_foot_min_clearance_to_small: Tensor | None = None
    base_small_penetration_count: Tensor | None = None
    base_min_clearance_to_small: Tensor | None = None
    base_path_crosses_small_flag: Tensor | None = None
    per_leg_touchdown_on_small_count: Tensor | None = None
    per_leg_foot_small_collision_count: Tensor | None = None
    per_leg_min_clearance_to_small: Tensor | None = None
    per_leg_touchdown_beyond_small_back_edge: Tensor | None = None
    touchdown_ground_gap_by_leg: Tensor | None = None
    touchdown_semantic_by_leg: Tensor | None = None
    touchdown_frame_by_leg: Tensor | None = None
    command_leading_before_trailing_schedule_ok: Tensor | None = None
    candidate_hard_reason_mask: Tensor | None = None
    selected_hard_reason_mask: Tensor | None = None
    candidate_hard_rank_cost: Tensor | None = None
    selected_hard_rank_cost: Tensor | None = None
    selected_candidate_index: Tensor | None = None

    def __post_init__(self) -> None:
        if self.root_pos.ndim != 3 or self.root_pos.shape[-1] != 3:
            raise ValueError("root_pos must have shape [B, T, 3]")
        if self.root_rpy.shape != self.root_pos.shape:
            raise ValueError("root_rpy must match root_pos")
        if self.foot_pos.ndim != 4 or self.foot_pos.shape[-2:] != (4, 3):
            raise ValueError("foot_pos must have shape [B, T, 4, 3]")
        if self.joint_angles.ndim != 3 or self.joint_angles.shape[-1] != 12:
            raise ValueError("joint_angles must have shape [B, T, 12]")
        if self.contact_state.ndim != 3 or self.contact_state.shape[-1] != 4:
            raise ValueError("contact_state must have shape [B, T, 4]")
        if self.touchdown_seq.ndim != 4 or self.touchdown_seq.shape[1:] != (4, 2, 3):
            raise ValueError("touchdown_seq must have shape [B, 4, 2, 3]")
        if self.touchdown_mask.ndim != 3 or self.touchdown_mask.shape[1:] != (4, 2):
            raise ValueError("touchdown_mask must have shape [B, 4, 2]")
        if self.selected_route_offset is not None:
            if self.selected_route_offset.ndim != 1 or self.selected_route_offset.shape[0] != self.root_pos.shape[0]:
                raise ValueError("selected_route_offset must have shape [B]")
        if self.state_mode is not None:
            if self.state_mode.ndim != 1 or self.state_mode.shape[0] != self.root_pos.shape[0]:
                raise ValueError("state_mode must have shape [B]")
        if self.small_strategy_outcome is not None:
            if self.small_strategy_outcome.ndim != 1 or self.small_strategy_outcome.shape[0] != self.root_pos.shape[0]:
                raise ValueError("small_strategy_outcome must have shape [B]")
        if self.semantic_candidate_costs is not None:
            if self.semantic_candidate_costs.ndim != 2 or self.semantic_candidate_costs.shape != (self.root_pos.shape[0], T116_CANDIDATE_COUNT):
                raise ValueError("semantic_candidate_costs must have shape [B, 5]")
        _validate_optional_shape(self.mode, (self.root_pos.shape[0],), "mode")
        _validate_optional_shape(self.selected_beta, (self.root_pos.shape[0],), "selected_beta")
        _validate_optional_shape(self.selected_route, (self.root_pos.shape[0],), "selected_route")
        _validate_optional_shape(self.direction_id, (self.root_pos.shape[0],), "direction_id")
        _validate_optional_shape(self.small_front_s, (self.root_pos.shape[0],), "small_front_s")
        _validate_optional_shape(self.small_back_s, (self.root_pos.shape[0],), "small_back_s")
        _validate_optional_shape(self.small_top_z, (self.root_pos.shape[0],), "small_top_z")
        _validate_optional_shape(self.command_direction_violation, (self.root_pos.shape[0],), "command_direction_violation")
        _validate_optional_shape(self.cross_small_success, (self.root_pos.shape[0],), "cross_small_success")
        if self.body_min_clearance is not None:
            if self.body_min_clearance.ndim != 1 or self.body_min_clearance.shape[0] != self.root_pos.shape[0]:
                raise ValueError("body_min_clearance must have shape [B]")
        if self.leg_min_clearance is not None:
            if self.leg_min_clearance.ndim != 1 or self.leg_min_clearance.shape[0] != self.root_pos.shape[0]:
                raise ValueError("leg_min_clearance must have shape [B]")
        if self.front_touchdown_ground_gap is not None:
            if self.front_touchdown_ground_gap.ndim != 2 or self.front_touchdown_ground_gap.shape != (self.root_pos.shape[0], 2):
                raise ValueError("front_touchdown_ground_gap must have shape [B, 2]")
        if self.rear_touchdown_ground_gap is not None:
            if self.rear_touchdown_ground_gap.ndim != 2 or self.rear_touchdown_ground_gap.shape != (self.root_pos.shape[0], 2):
                raise ValueError("rear_touchdown_ground_gap must have shape [B, 2]")
        if self.touchdown_on_small_count is not None:
            if self.touchdown_on_small_count.ndim != 1 or self.touchdown_on_small_count.shape[0] != self.root_pos.shape[0]:
                raise ValueError("touchdown_on_small_count must have shape [B]")
        if self.front_foot_small_collision_count is not None:
            if self.front_foot_small_collision_count.ndim != 1 or self.front_foot_small_collision_count.shape[0] != self.root_pos.shape[0]:
                raise ValueError("front_foot_small_collision_count must have shape [B]")
        if self.rear_foot_small_collision_count is not None:
            if self.rear_foot_small_collision_count.ndim != 1 or self.rear_foot_small_collision_count.shape[0] != self.root_pos.shape[0]:
                raise ValueError("rear_foot_small_collision_count must have shape [B]")
        if self.front_foot_min_clearance_to_small is not None:
            if self.front_foot_min_clearance_to_small.ndim != 1 or self.front_foot_min_clearance_to_small.shape[0] != self.root_pos.shape[0]:
                raise ValueError("front_foot_min_clearance_to_small must have shape [B]")
        if self.rear_foot_min_clearance_to_small is not None:
            if self.rear_foot_min_clearance_to_small.ndim != 1 or self.rear_foot_min_clearance_to_small.shape[0] != self.root_pos.shape[0]:
                raise ValueError("rear_foot_min_clearance_to_small must have shape [B]")
        if self.base_small_penetration_count is not None:
            if self.base_small_penetration_count.ndim != 1 or self.base_small_penetration_count.shape[0] != self.root_pos.shape[0]:
                raise ValueError("base_small_penetration_count must have shape [B]")
        if self.base_min_clearance_to_small is not None:
            if self.base_min_clearance_to_small.ndim != 1 or self.base_min_clearance_to_small.shape[0] != self.root_pos.shape[0]:
                raise ValueError("base_min_clearance_to_small must have shape [B]")
        if self.base_path_crosses_small_flag is not None:
            if self.base_path_crosses_small_flag.ndim != 1 or self.base_path_crosses_small_flag.shape[0] != self.root_pos.shape[0]:
                raise ValueError("base_path_crosses_small_flag must have shape [B]")
        _validate_optional_shape(self.per_leg_touchdown_on_small_count, (self.root_pos.shape[0], 4), "per_leg_touchdown_on_small_count")
        _validate_optional_shape(self.per_leg_foot_small_collision_count, (self.root_pos.shape[0], 4), "per_leg_foot_small_collision_count")
        _validate_optional_shape(self.per_leg_min_clearance_to_small, (self.root_pos.shape[0], 4), "per_leg_min_clearance_to_small")
        _validate_optional_shape(self.per_leg_touchdown_beyond_small_back_edge, (self.root_pos.shape[0], 4), "per_leg_touchdown_beyond_small_back_edge")
        _validate_optional_shape(self.touchdown_ground_gap_by_leg, (self.root_pos.shape[0], 4), "touchdown_ground_gap_by_leg")
        _validate_optional_shape(self.touchdown_semantic_by_leg, (self.root_pos.shape[0], 4), "touchdown_semantic_by_leg")
        _validate_optional_shape(self.touchdown_frame_by_leg, (self.root_pos.shape[0], 4), "touchdown_frame_by_leg")
        _validate_optional_shape(
            self.command_leading_before_trailing_schedule_ok,
            (self.root_pos.shape[0],),
            "command_leading_before_trailing_schedule_ok",
        )
        _validate_optional_shape(
            self.candidate_hard_reason_mask,
            (self.root_pos.shape[0], T116_CANDIDATE_COUNT, HARD_REASON_COUNT),
            "candidate_hard_reason_mask",
        )
        _validate_optional_shape(self.selected_hard_reason_mask, (self.root_pos.shape[0], HARD_REASON_COUNT), "selected_hard_reason_mask")
        _validate_optional_shape(self.candidate_hard_rank_cost, (self.root_pos.shape[0], T116_CANDIDATE_COUNT), "candidate_hard_rank_cost")
        _validate_optional_shape(self.selected_hard_rank_cost, (self.root_pos.shape[0],), "selected_hard_rank_cost")
        _validate_optional_shape(self.selected_candidate_index, (self.root_pos.shape[0],), "selected_candidate_index")
        if self.candidate_hard_reason_mask is not None and self.candidate_hard_reason_mask.dtype != torch.bool:
            raise ValueError("candidate_hard_reason_mask must be bool")
        if self.selected_hard_reason_mask is not None and self.selected_hard_reason_mask.dtype != torch.bool:
            raise ValueError("selected_hard_reason_mask must be bool")


__all__ = [
    "CALF_LENGTH",
    "HARD_REASON_BASE_SMALL_PENETRATION",
    "HARD_REASON_BODY_HARD_COLLISION",
    "HARD_REASON_BOUNDARY_INVALID",
    "HARD_REASON_COUNT",
    "HARD_REASON_CROSSING_NOT_GROUNDED",
    "HARD_REASON_DIRECTION_VIOLATION",
    "HARD_REASON_FOOT_LARGE_COLLISION",
    "HARD_REASON_FRONT_FOOT_SMALL_COLLISION",
    "HARD_REASON_LEG_HARD_COLLISION",
    "HARD_REASON_NAMES",
    "HARD_REASON_PATH_COLLISION",
    "HARD_REASON_PER_LEG_FOOT_SMALL_COLLISION",
    "HARD_REASON_REAR_FOOT_SMALL_COLLISION",
    "HARD_REASON_TOUCHDOWN_ON_LARGE",
    "HARD_REASON_TOUCHDOWN_ON_SMALL",
    "HIP_HEIGHT",
    "HIP_OFFSETS_ARRAY",
    "HIP_OFFSET_Y",
    "THIGH_LENGTH",
    "T116_CANDIDATE_COUNT",
    "TOGETHER_TERRAIN_REFERENCE_STENCIL_RADIUS_M",
    "TOGETHER_TERRAIN_REFERENCE_STENCIL_STEP_M",
    "TogetherContactSchedule",
    "TogetherPlannerResult",
    "TogetherPlannerStatus",
    "TogetherRobotState",
    "TogetherTerrainSemanticId",
]
