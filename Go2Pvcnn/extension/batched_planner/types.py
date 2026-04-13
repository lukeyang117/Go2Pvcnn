"""Batched planner data types mirroring raw go2fp conventions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

GRAVITY = 9.81
HIP_HEIGHT = 0.30

LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_FRONT_SIGN = {"FL": 1.0, "FR": 1.0, "RL": -1.0, "RR": -1.0}
LEG_SIDE_SIGN = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}

HIP_OFFSETS = {
    "FL": torch.tensor([0.1934, 0.0465, 0.0], dtype=torch.float32),
    "FR": torch.tensor([0.1934, -0.0465, 0.0], dtype=torch.float32),
    "RL": torch.tensor([-0.1934, 0.0465, 0.0], dtype=torch.float32),
    "RR": torch.tensor([-0.1934, -0.0465, 0.0], dtype=torch.float32),
}
HIP_OFFSETS_ARRAY = torch.tensor(
    [
        [0.1934, 0.0465, 0.0],
        [0.1934, -0.0465, 0.0],
        [-0.1934, 0.0465, 0.0],
        [-0.1934, -0.0465, 0.0],
    ],
    dtype=torch.float32,
)

THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_OFFSET_Y = 0.0955
MASS = 15.0


@dataclass(frozen=True)
class BatchedRobotState:
    root_pos: Tensor
    root_quat: Tensor
    joint_angles: Tensor
    foot_pos: Tensor
    foot_vel: Tensor | None = None

    def __post_init__(self) -> None:
        if self.foot_vel is None:
            object.__setattr__(self, "foot_vel", torch.zeros_like(self.foot_pos))


@dataclass(frozen=True)
class BatchedTrajectoryResult:
    num_frames: int
    root_pos_w: Tensor
    root_quat_w: Tensor
    root_lin_vel_w: Tensor
    root_ang_vel_w: Tensor
    joint_angles: Tensor
    foot_pos_w: Tensor
    foot_pos_root: Tensor
    contact_state: Tensor
    body_pos_root: Tensor
    planned_touchdown_w: Tensor


__all__ = [
    "BatchedRobotState",
    "BatchedTrajectoryResult",
    "CALF_LENGTH",
    "GRAVITY",
    "HIP_HEIGHT",
    "HIP_OFFSETS",
    "HIP_OFFSETS_ARRAY",
    "HIP_OFFSET_Y",
    "LEG_FRONT_SIGN",
    "LEG_ORDER",
    "LEG_SIDE_SIGN",
    "MASS",
    "THIGH_LENGTH",
]
