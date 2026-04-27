"""Native batched-together planner tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

HIP_HEIGHT = 0.30
HIP_OFFSET_Y = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_OFFSETS_ARRAY = torch.tensor(
    (
        (0.1934, 0.0465, 0.0),
        (0.1934, -0.0465, 0.0),
        (-0.1934, 0.0465, 0.0),
        (-0.1934, -0.0465, 0.0),
    ),
    dtype=torch.float32,
)


class TogetherPlannerStatus(IntEnum):
    OK = 0
    ALL_INFEASIBLE = 1
    INVALID_CONFIG = 2


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


__all__ = [
    "CALF_LENGTH",
    "HIP_HEIGHT",
    "HIP_OFFSETS_ARRAY",
    "HIP_OFFSET_Y",
    "THIGH_LENGTH",
    "TogetherContactSchedule",
    "TogetherPlannerResult",
    "TogetherPlannerStatus",
    "TogetherRobotState",
]
