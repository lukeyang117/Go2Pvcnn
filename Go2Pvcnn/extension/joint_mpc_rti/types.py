"""Fixed-shape tensor contracts for joint MPC RTI."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def _require_shape(name: str, tensor: Tensor, suffix: tuple[int, ...]) -> None:
    if tensor.ndim != len(suffix) + 1 or tuple(tensor.shape[1:]) != suffix:
        raise ValueError(f"{name} must have shape [B,{','.join(str(value) for value in suffix)}]")


@dataclass(frozen=True)
class JointMpcRtiState:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    root_lin_vel_b: Tensor
    root_ang_vel_b: Tensor
    joint_vel: Tensor

    def __post_init__(self) -> None:
        _require_shape("root_pos_w", self.root_pos_w, (3,))
        _require_shape("root_rpy_w", self.root_rpy_w, (3,))
        _require_shape("joint_pos", self.joint_pos, (12,))
        _require_shape("root_lin_vel_b", self.root_lin_vel_b, (3,))
        _require_shape("root_ang_vel_b", self.root_ang_vel_b, (3,))
        _require_shape("joint_vel", self.joint_vel, (12,))
        batch = int(self.root_pos_w.shape[0])
        tensors = (
            self.root_rpy_w,
            self.joint_pos,
            self.root_lin_vel_b,
            self.root_ang_vel_b,
            self.joint_vel,
        )
        if any(int(tensor.shape[0]) != batch for tensor in tensors):
            raise ValueError("all state tensors must share the same batch dimension")
        if any(tensor.device != self.root_pos_w.device for tensor in tensors):
            raise ValueError("all state tensors must share the same device")
        if any(tensor.dtype != self.root_pos_w.dtype for tensor in tensors):
            raise ValueError("all state tensors must share the same dtype")

    @property
    def batch_size(self) -> int:
        return int(self.root_pos_w.shape[0])

    @property
    def device(self) -> torch.device:
        return self.root_pos_w.device

    def as_vector(self) -> Tensor:
        return torch.cat((self.root_pos_w, self.root_rpy_w, self.joint_pos), dim=-1)


@dataclass(frozen=True)
class JointMpcRtiTrajectory:
    state: Tensor
    control: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    valid: Tensor
    fallback: Tensor
    status: Tensor
    loss_breakdown: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class JointMpcRtiStepResult:
    full_trajectory: JointMpcRtiTrajectory
    pending_reference: object | None
    solver_state: object | None


@dataclass(frozen=True)
class JointMpcRtiSolverState:
    state: Tensor
    control: Tensor
    dual: Tensor | None
    previous_control: Tensor
    gait_phase: Tensor | None = None
    stance_anchor_w: Tensor | None = None


@dataclass(frozen=True)
class JointMpcPendingReference:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_angles: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    valid: Tensor
    target_step: int = 1


@dataclass(frozen=True)
class JointMpcTerrainField:
    height_w: Tensor
    semantic_id: Tensor
    small_distance_m: Tensor
    large_distance_m: Tensor
    small_gradient_xy: Tensor
    large_gradient_xy: Tensor
    valid_mask: Tensor
    origin_w: Tensor
    yaw_w: Tensor
    timestamp: Tensor
    version: Tensor
    resolution: float


__all__ = [
    "JointMpcRtiState",
    "JointMpcRtiSolverState",
    "JointMpcRtiStepResult",
    "JointMpcRtiTrajectory",
    "JointMpcPendingReference",
    "JointMpcTerrainField",
]
