"""Convention helpers for the batched GPU kinematic planner.

This module keeps the planner-facing convention in wxyz order while providing
Isaac-facing xyzw conversions at the boundary.
"""

from __future__ import annotations

import torch
from torch import Tensor


def quat_wxyz_to_xyzw(q: Tensor) -> Tensor:
    """Convert a quaternion from wxyz order to xyzw order."""
    q = torch.as_tensor(q)
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


def quat_xyzw_to_wxyz(q: Tensor) -> Tensor:
    """Convert a quaternion from xyzw order to wxyz order."""
    q = torch.as_tensor(q)
    return torch.cat([q[..., -1:], q[..., :3]], dim=-1)


def extract_yaw_batch(quat_wxyz: Tensor) -> Tensor:
    """Return yaw angles from wxyz quaternions."""
    q = torch.as_tensor(quat_wxyz)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def extract_roll_pitch_batch(quat_wxyz: Tensor) -> tuple[Tensor, Tensor]:
    """Return roll and pitch from wxyz quaternions."""
    q = torch.as_tensor(quat_wxyz)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    return roll, pitch


def euler_to_quat_batch(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """Convert batched ZYX Euler angles to wxyz quaternions."""
    roll = torch.as_tensor(roll)
    pitch = torch.as_tensor(pitch)
    yaw = torch.as_tensor(yaw)
    hr = 0.5 * roll
    hp = 0.5 * pitch
    hy = 0.5 * yaw
    cr, sr = torch.cos(hr), torch.sin(hr)
    cp, sp = torch.cos(hp), torch.sin(hp)
    cy, sy = torch.cos(hy), torch.sin(hy)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def yaw_rotation_matrix_batch(yaw: Tensor) -> Tensor:
    """Return batched +Z yaw rotation matrices."""
    yaw = torch.as_tensor(yaw)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    row0 = torch.stack([c, -s, torch.zeros_like(c)], dim=-1)
    row1 = torch.stack([s, c, torch.zeros_like(c)], dim=-1)
    row2 = torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def isaac_state_to_planner_state(
    root_pos_w: Tensor,
    root_quat_xyzw: Tensor,
    joint_pos: Tensor,
    foot_pos_w: Tensor,
    foot_vel_w: Tensor | None = None,
):
    """Convert Isaac-facing state tensors into planner-facing batched state."""
    from .batched_planner.types import BatchedRobotState

    foot_pos = torch.as_tensor(foot_pos_w)
    return BatchedRobotState(
        root_pos=torch.as_tensor(root_pos_w),
        root_quat=quat_xyzw_to_wxyz(root_quat_xyzw),
        joint_angles=torch.as_tensor(joint_pos),
        foot_pos=foot_pos,
        foot_vel=foot_pos.new_zeros(foot_pos.shape) if foot_vel_w is None else torch.as_tensor(foot_vel_w),
    )


def _normalize_planner_result_for_reference_cache(result):
    """Canonicalize planner output tensors for the reference cache ABI."""
    from .reference.cache import (
        CANONICAL_REFERENCE_CACHE_DEVICE,
        canonical_reference_cache_bool_tensor,
        canonical_reference_cache_float_tensor,
        canonical_reference_cache_index_tensor,
    )

    root_pos_w = canonical_reference_cache_float_tensor(result.root_pos_w)
    root_quat_w = canonical_reference_cache_float_tensor(result.root_quat_w)
    root_lin_vel_w = canonical_reference_cache_float_tensor(result.root_lin_vel_w)
    root_ang_vel_w = canonical_reference_cache_float_tensor(result.root_ang_vel_w)
    joint_angles = canonical_reference_cache_float_tensor(result.joint_angles)
    foot_pos_w = canonical_reference_cache_float_tensor(result.foot_pos_w)
    foot_pos_root = canonical_reference_cache_float_tensor(result.foot_pos_root)
    contact_state = canonical_reference_cache_bool_tensor(result.contact_state)
    body_pos_root = canonical_reference_cache_float_tensor(result.body_pos_root)

    tensors = {
        "root_pos_w": root_pos_w,
        "root_quat_w": root_quat_w,
        "root_lin_vel_w": root_lin_vel_w,
        "root_ang_vel_w": root_ang_vel_w,
        "joint_angles": joint_angles,
        "foot_pos_w": foot_pos_w,
        "foot_pos_root": foot_pos_root,
        "contact_state": contact_state,
        "body_pos_root": body_pos_root,
    }
    num_envs = int(root_pos_w.shape[0])
    num_frames = int(root_pos_w.shape[1])
    for name, tensor in tensors.items():
        if tensor.shape[0] != num_envs:
            raise ValueError(f"{name} batch size mismatch: expected {num_envs}, got {tensor.shape[0]}")
        if tensor.ndim < 2 or tensor.shape[1] != num_frames:
            raise ValueError(
                f"{name} num_frames mismatch: expected {num_frames}, got {tensor.shape[1] if tensor.ndim >= 2 else 'n/a'}"
            )
    if int(result.num_frames) != num_frames:
        raise ValueError(
            f"result.num_frames mismatch: expected {num_frames} from tensors, got {int(result.num_frames)}"
        )

    device = CANONICAL_REFERENCE_CACHE_DEVICE
    phase_index = canonical_reference_cache_index_tensor(torch.arange(num_frames, device=device))
    phase_index = phase_index.unsqueeze(0).expand(num_envs, -1).clone()
    valid_mask = canonical_reference_cache_bool_tensor(torch.ones_like(phase_index, dtype=torch.bool))
    planned_touchdown_w = canonical_reference_cache_float_tensor(result.planned_touchdown_w)
    if planned_touchdown_w.ndim == 2:
        planned_touchdown_w = planned_touchdown_w.reshape(1, 1, 4, 3).expand(num_envs, num_frames, 4, 3).clone()
    elif planned_touchdown_w.ndim == 3:
        planned_touchdown_w = planned_touchdown_w.unsqueeze(1).expand(num_envs, num_frames, 4, 3).clone()
    elif planned_touchdown_w.ndim == 4:
        if planned_touchdown_w.shape[0] != num_envs or planned_touchdown_w.shape[1] != num_frames:
            raise ValueError(
                f"planned_touchdown_w batch/frame mismatch: expected {(num_envs, num_frames)}, "
                f"got {tuple(planned_touchdown_w.shape[:2])}"
            )
    else:
        raise ValueError(f"planned_touchdown_w must have shape (N,4,3), (N,H,4,3), or (4,3); got {tuple(planned_touchdown_w.shape)}")
    return {
        "root_pos_w": root_pos_w,
        "root_quat_w": root_quat_w,
        "root_lin_vel_w": root_lin_vel_w,
        "root_ang_vel_w": root_ang_vel_w,
        "joint_angles": joint_angles,
        "foot_pos_root": foot_pos_root,
        "contact_state": contact_state,
        "planned_touchdown_w": planned_touchdown_w,
        "phase_index": phase_index,
        "valid_mask": valid_mask,
    }


def planner_result_to_reference_cache(result):
    """Convert planner output into the canonical reference cache ABI."""
    from .reference.cache import ReferenceTrajectoryCache

    normalized = _normalize_planner_result_for_reference_cache(result)
    return ReferenceTrajectoryCache(
        root_pos_w=normalized["root_pos_w"],
        root_quat_w=normalized["root_quat_w"],
        joint_angles=normalized["joint_angles"],
        foot_pos_root=normalized["foot_pos_root"],
        contact_state=normalized["contact_state"],
        planned_touchdown_w=normalized["planned_touchdown_w"],
        phase_index=normalized["phase_index"],
        valid_mask=normalized["valid_mask"],
    )


__all__ = [
    "euler_to_quat_batch",
    "extract_roll_pitch_batch",
    "extract_yaw_batch",
    "isaac_state_to_planner_state",
    "planner_result_to_reference_cache",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "yaw_rotation_matrix_batch",
]
