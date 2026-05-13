"""Tensor contracts for the batch MPC backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

MPC_HARD_REASON_NAMES = (
    "large_collision",
    "small_collision_or_clearance",
    "touchdown_unsupported",
    "ik_workspace_impossible",
    "airborne_support_instability",
    "command_progress_violation",
)
MPC_HARD_REASON_COUNT = len(MPC_HARD_REASON_NAMES)


class MpcPlannerStatus(IntEnum):
    OK = 0
    ALL_INFEASIBLE = 1
    INVALID_CONFIG = 2


@dataclass(frozen=True)
class MpcRobotState:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    joint_angles: Tensor
    foot_vel: Tensor | None = None


@dataclass(frozen=True)
class MpcFootholdMemory:
    foot_rel_body_seed: Tensor | None = None
    stance_anchor_w: Tensor | None = None
    yaw_entry_ramp: Tensor | None = None


@dataclass(frozen=True)
class MpcPlannerTerrain:
    height_map: Tensor
    world_x_range: tuple[float, float]
    world_y_range: tuple[float, float]
    semantic_map: Tensor | None = None


@dataclass(frozen=True)
class MpcPlannerResult:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    joint_angles: Tensor
    contact_state: Tensor
    touchdown_seq: Tensor
    planned_touchdown_w: Tensor
    cost_total: Tensor
    cost_breakdown: dict[str, Tensor]
    status: Tensor
    feasible: Tensor
    safe_fallback: Tensor
    loss_breakdown: dict[str, Tensor] | None = None
    hard_reason_mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.root_pos.ndim != 3 or int(self.root_pos.shape[-1]) != 3:
            raise ValueError("root_pos must have shape [B, T, 3]")
        if self.root_rpy.shape != self.root_pos.shape:
            raise ValueError("root_rpy must match root_pos shape")
        if self.foot_pos.ndim != 4 or tuple(self.foot_pos.shape[-2:]) != (4, 3):
            raise ValueError("foot_pos must have shape [B, T, 4, 3]")
        if self.joint_angles.ndim != 3 or int(self.joint_angles.shape[-1]) != 12:
            raise ValueError("joint_angles must have shape [B, T, 12]")
        if self.contact_state.ndim != 3 or int(self.contact_state.shape[-1]) != 4:
            raise ValueError("contact_state must have shape [B, T, 4]")
        if self.touchdown_seq.ndim != 4 or int(self.touchdown_seq.shape[1]) != 4 or int(self.touchdown_seq.shape[-1]) != 3:
            raise ValueError("touchdown_seq must have shape [B, 4, E, 3]")
        if self.planned_touchdown_w.ndim != 4 or tuple(self.planned_touchdown_w.shape[-2:]) != (4, 3):
            raise ValueError("planned_touchdown_w must have shape [B, T, 4, 3]")
        if self.status.ndim != 1 or int(self.status.shape[0]) != int(self.root_pos.shape[0]):
            raise ValueError("status must have shape [B]")
        if self.feasible.ndim != 1 or int(self.feasible.shape[0]) != int(self.root_pos.shape[0]):
            raise ValueError("feasible must have shape [B]")
        if self.safe_fallback.ndim != 1 or int(self.safe_fallback.shape[0]) != int(self.root_pos.shape[0]):
            raise ValueError("safe_fallback must have shape [B]")
        if self.hard_reason_mask is not None:
            expected = (int(self.root_pos.shape[0]), MPC_HARD_REASON_COUNT)
            if tuple(self.hard_reason_mask.shape) != expected:
                raise ValueError(f"hard_reason_mask must have shape {expected}")
            if self.hard_reason_mask.dtype != torch.bool:
                raise ValueError("hard_reason_mask must be bool")


__all__ = [
    "MPC_HARD_REASON_COUNT",
    "MPC_HARD_REASON_NAMES",
    "MpcFootholdMemory",
    "MpcPlannerResult",
    "MpcPlannerStatus",
    "MpcPlannerTerrain",
    "MpcRobotState",
]
