from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_FRONT_SIGN = {"FL": 1.0, "FR": 1.0, "RL": -1.0, "RR": -1.0}
LEG_SIDE_SIGN = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}

HIP_OFFSETS = {
    "FL": np.array([0.1934, 0.0465, 0.0], dtype=np.float64),
    "FR": np.array([0.1934, -0.0465, 0.0], dtype=np.float64),
    "RL": np.array([-0.1934, 0.0465, 0.0], dtype=np.float64),
    "RR": np.array([-0.1934, -0.0465, 0.0], dtype=np.float64),
}
HIP_OFFSETS_ARRAY = np.array(
    [
        [0.1934, 0.0465, 0.0],
        [0.1934, -0.0465, 0.0],
        [-0.1934, 0.0465, 0.0],
        [-0.1934, -0.0465, 0.0],
    ],
    dtype=np.float64,
)

THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213
HIP_OFFSET_Y = 0.0955
HIP_HEIGHT = 0.30
MASS = 15.0
GRAVITY = 9.81


@dataclass(frozen=True)
class RobotState:
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_angles: np.ndarray
    foot_pos: np.ndarray
    foot_vel: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 3), dtype=np.float64)
    )


@dataclass(frozen=True)
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class TrajectoryResult:
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_angles: np.ndarray
    foot_pos_w: np.ndarray
    foot_pos_root: np.ndarray
    contact_state: np.ndarray
    body_pos_root: np.ndarray
    planned_touchdown_w: np.ndarray


def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    row0 = np.stack([r00, r01, r02], axis=-1)
    row1 = np.stack([r10, r11, r12], axis=-1)
    row2 = np.stack([r20, r21, r22], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX roll-pitch-yaw angles to a wxyz quaternion."""
    hr, hp, hy = 0.5 * roll, 0.5 * pitch, 0.5 * yaw
    cr, sr = np.cos(hr), np.sin(hr)
    cp, sp = np.cos(hp), np.sin(hp)
    cy, sy = np.cos(hy), np.sin(hy)
    qx = np.array([cr, sr, 0.0, 0.0], dtype=np.float64)
    qy = np.array([cp, 0.0, sp, 0.0], dtype=np.float64)
    qz = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)
    return quat_multiply(qz, quat_multiply(qy, qx))


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Create a pure yaw rotation quaternion in wxyz order."""
    hy = 0.5 * yaw
    return np.array([np.cos(hy), 0.0, 0.0, np.sin(hy)], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product with wxyz quaternion convention."""
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = np.split(a, 4, axis=-1)
    w2, x2, y2, z2 = np.split(b, 4, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Return the inverse of a unit quaternion."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)
    out[..., 0] = q[..., 0]
    out[..., 1:] = -q[..., 1:]
    return out


def rotate_vector(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a vector or batch of vectors by quaternion q."""
    R = quat_to_rot_matrix(q)
    v = np.asarray(v, dtype=np.float64)
    if R.ndim == 2:
        if v.ndim == 1:
            return R @ v
        return np.einsum("ij,nj->ni", R, v)
    if v.ndim == 1:
        return np.einsum("nij,j->ni", R, v)
    return np.einsum("nij,nj->ni", R, v)


def rotate_vector_inverse(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a vector or batch of vectors by the inverse quaternion."""
    R = quat_to_rot_matrix(q)
    Rt = np.swapaxes(R, -1, -2)
    v = np.asarray(v, dtype=np.float64)
    if Rt.ndim == 2:
        if v.ndim == 1:
            return Rt @ v
        return np.einsum("ij,nj->ni", Rt, v)
    if v.ndim == 1:
        return np.einsum("nij,j->ni", Rt, v)
    return np.einsum("nij,nj->ni", Rt, v)


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    """Return a right-handed +Z yaw rotation matrix."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


__all__ = [
    "CALF_LENGTH",
    "Command",
    "GRAVITY",
    "HIP_HEIGHT",
    "HIP_OFFSETS",
    "HIP_OFFSETS_ARRAY",
    "HIP_OFFSET_Y",
    "LEG_FRONT_SIGN",
    "LEG_ORDER",
    "LEG_SIDE_SIGN",
    "MASS",
    "RobotState",
    "THIGH_LENGTH",
    "TrajectoryResult",
    "euler_to_quat",
    "quat_from_yaw",
    "quat_inverse",
    "quat_multiply",
    "quat_to_rot_matrix",
    "rotate_vector",
    "rotate_vector_inverse",
    "yaw_rotation_matrix",
]
