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
    derived_velocity: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    valid: Tensor
    fallback: Tensor
    status: Tensor
    line_search_alpha: Tensor
    loss_breakdown: dict[str, Tensor] = field(default_factory=dict)
    cold_start: Tensor | None = None
    warm_start: Tensor | None = None
    warm_cache_invariant_fault: Tensor | None = None
    state_nodes: Tensor | None = None
    future_state: Tensor | None = None
    publish: Tensor | None = None
    stop: Tensor | None = None

    def __post_init__(self) -> None:
        nodes = self.state if self.state_nodes is None else self.state_nodes
        if nodes.ndim != 3 or tuple(nodes.shape[1:]) != (31, 18):
            raise ValueError("state_nodes must have shape [B,31,18]")
        future = nodes[:, 1:] if self.future_state is None else self.future_state
        if future.ndim != 3 or tuple(future.shape[1:]) != (30, 18):
            raise ValueError("future_state must have shape [B,30,18]")
        if int(future.shape[0]) != int(nodes.shape[0]):
            raise ValueError("state_nodes and future_state must share the batch dimension")
        object.__setattr__(self, "state_nodes", nodes)
        object.__setattr__(self, "future_state", future)
        object.__setattr__(self, "publish", self.valid if self.publish is None else self.publish)
        object.__setattr__(self, "stop", ~self.valid if self.stop is None else self.stop)


@dataclass(frozen=True)
class JointMpcRtiStepDiagnostics:
    nominal_state: Tensor
    qp_direction: Tensor
    stance_anchor_w: Tensor
    touchdown_reference_w: Tensor
    candidate_loss: Tensor
    candidate_filter_valid: Tensor
    candidate_swing_safe_z: Tensor
    support_target: Tensor
    node_loss_breakdown: dict[str, Tensor]


@dataclass(frozen=True)
class JointMpcRtiStepResult:
    full_trajectory: JointMpcRtiTrajectory
    pending_reference: object | None
    solver_state: object | None
    diagnostics: JointMpcRtiStepDiagnostics | None = None


@dataclass(frozen=True)
class JointMpcRtiSolverState:
    trajectory: Tensor
    gait_phase: Tensor
    initialized: Tensor
    stance_anchor_w: Tensor


@dataclass(frozen=True)
class JointMpcFieldFrame:
    origin_w: Tensor
    yaw_w: Tensor
    timestamp: Tensor
    refresh_id: Tensor

    def __post_init__(self) -> None:
        _require_shape("origin_w", self.origin_w, (3,))
        batch = int(self.origin_w.shape[0])
        for name, tensor in (
            ("yaw_w", self.yaw_w),
            ("timestamp", self.timestamp),
            ("refresh_id", self.refresh_id),
        ):
            if tensor.shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")


@dataclass(frozen=True)
class JointMpcPerceptiveField:
    height_w: Tensor
    semantic_id: Tensor
    valid_mask: Tensor
    small_mask: Tensor
    large_mask: Tensor
    unknown_mask: Tensor
    inflated_height_w: Tensor
    landing_safe: Tensor
    slope_xy: Tensor
    slope_rad: Tensor
    roughness: Tensor
    semantic_edge_mask: Tensor
    origin_w: Tensor
    yaw_w: Tensor
    timestamp: Tensor
    refresh_id: Tensor
    resolution: float


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
    small_occupancy: Tensor | None = None
    large_occupancy: Tensor | None = None
    small_propagated_height: Tensor | None = None
    large_propagated_height: Tensor | None = None
    small_occupancy_gradient_xy: Tensor | None = None
    large_occupancy_gradient_xy: Tensor | None = None


__all__ = [
    "JointMpcFieldFrame",
    "JointMpcPerceptiveField",
    "JointMpcRtiState",
    "JointMpcRtiSolverState",
    "JointMpcRtiStepResult",
    "JointMpcRtiTrajectory",
    "JointMpcPendingReference",
    "JointMpcTerrainField",
]
