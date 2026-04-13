"""Pure NumPy inverse/forward kinematics helpers for the Go2 planner core."""

from __future__ import annotations

import numpy as np

from .types import (
    CALF_LENGTH,
    HIP_OFFSET_Y,
    HIP_OFFSETS_ARRAY,
    LEG_ORDER,
    LEG_SIDE_SIGN,
    THIGH_LENGTH,
)

JOINT_LIMITS = np.array(
    [
        [-1.0472, 1.0472],  # FL hip
        [-1.5708, 3.4907],  # FL thigh
        [-2.7227, -0.8378],  # FL calf
        [-1.0472, 1.0472],  # FR hip
        [-1.5708, 3.4907],  # FR thigh
        [-2.7227, -0.8378],  # FR calf
        [-1.0472, 1.0472],  # RL hip
        [-0.5236, 4.5379],  # RL thigh
        [-2.7227, -0.8378],  # RL calf
        [-1.0472, 1.0472],  # RR hip
        [-0.5236, 4.5379],  # RR thigh
        [-2.7227, -0.8378],  # RR calf
    ],
    dtype=np.float64,
)


def _quat_to_rot_batch(quat: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions to rotation matrices."""
    q = np.asarray(quat, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[..., 0, 1] = 2.0 * (xy - wz)
    R[..., 0, 2] = 2.0 * (xz + wy)
    R[..., 1, 0] = 2.0 * (xy + wz)
    R[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[..., 1, 2] = 2.0 * (yz - wx)
    R[..., 2, 0] = 2.0 * (xz - wy)
    R[..., 2, 1] = 2.0 * (yz + wx)
    R[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def solve_leg_ik(
    foot_pos_hip: np.ndarray,
    side_sign: float,
    thigh_length: float = THIGH_LENGTH,
    calf_length: float = CALF_LENGTH,
    hip_offset_y: float = HIP_OFFSET_Y,
) -> np.ndarray:
    """Analytical IK for a single leg in the Go2 joint convention."""
    px = float(foot_pos_hip[0])
    py = float(foot_pos_hip[1])
    pz = float(foot_pos_hip[2])
    d = hip_offset_y * side_sign

    yz_sq = py * py + pz * pz
    lateral = np.sqrt(max(yz_sq - d * d, 0.0))
    hip_angle = np.arctan2(py, -pz) - np.arctan2(d, lateral)

    px_eff = px
    pz_eff = -lateral
    reach_sq = px_eff * px_eff + pz_eff * pz_eff
    cos_calf = (reach_sq - thigh_length**2 - calf_length**2) / (
        2.0 * thigh_length * calf_length
    )
    cos_calf = float(np.clip(cos_calf, -1.0, 1.0))
    calf_angle = -np.arccos(cos_calf)

    alpha = np.arctan2(-px_eff, -pz_eff)
    beta = np.arctan2(
        calf_length * np.sin(calf_angle),
        thigh_length + calf_length * np.cos(calf_angle),
    )
    thigh_angle = alpha - beta
    return np.array([hip_angle, thigh_angle, calf_angle], dtype=np.float64)


def _solve_leg_ik_batch(
    foot_pos_hip: np.ndarray,
    side_sign: float,
    thigh_length: float = THIGH_LENGTH,
    calf_length: float = CALF_LENGTH,
    hip_offset_y: float = HIP_OFFSET_Y,
) -> np.ndarray:
    """Vectorised analytical IK for one leg across N frames."""
    px = foot_pos_hip[:, 0]
    py = foot_pos_hip[:, 1]
    pz = foot_pos_hip[:, 2]
    d = hip_offset_y * side_sign

    yz_sq = py * py + pz * pz
    lateral = np.sqrt(np.clip(yz_sq - d * d, 0.0, None))
    hip_angle = np.arctan2(py, -pz) - np.arctan2(d, lateral)

    px_eff = px
    pz_eff = -lateral
    reach_sq = px_eff * px_eff + pz_eff * pz_eff
    cos_calf = (reach_sq - thigh_length**2 - calf_length**2) / (
        2.0 * thigh_length * calf_length
    )
    cos_calf = np.clip(cos_calf, -1.0, 1.0)
    calf_angle = -np.arccos(cos_calf)

    alpha = np.arctan2(-px_eff, -pz_eff)
    beta = np.arctan2(
        calf_length * np.sin(calf_angle),
        thigh_length + calf_length * np.cos(calf_angle),
    )
    thigh_angle = alpha - beta
    return np.stack([hip_angle, thigh_angle, calf_angle], axis=-1)


def forward_kinematics_leg(
    joint_angles: np.ndarray,
    side_sign: float,
    hip_offset: np.ndarray,
    thigh_length: float = THIGH_LENGTH,
    calf_length: float = CALF_LENGTH,
    hip_offset_y: float = HIP_OFFSET_Y,
) -> dict:
    """Forward kinematics for a single leg, returning body-frame positions."""
    h = float(joint_angles[0])
    theta_t = float(joint_angles[1])
    theta_c = float(joint_angles[2])
    d = hip_offset_y * side_sign

    cos_h, sin_h = np.cos(h), np.sin(h)

    knee_x = -thigh_length * np.sin(theta_t)
    knee_z = -thigh_length * np.cos(theta_t)

    calf_abs = theta_t + theta_c
    foot_x = knee_x - calf_length * np.sin(calf_abs)
    foot_z = knee_z - calf_length * np.cos(calf_abs)

    knee_local = np.array([knee_x, d, knee_z], dtype=np.float64)
    foot_local = np.array([foot_x, d, foot_z], dtype=np.float64)

    def _rx(v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                v[0],
                cos_h * v[1] - sin_h * v[2],
                sin_h * v[1] + cos_h * v[2],
            ],
            dtype=np.float64,
        )

    return {
        "hip": np.asarray(hip_offset, dtype=np.float64).copy(),
        "knee": np.asarray(hip_offset, dtype=np.float64) + _rx(knee_local),
        "foot": np.asarray(hip_offset, dtype=np.float64) + _rx(foot_local),
    }


def _forward_kinematics_leg_batch(
    angles: np.ndarray,
    side_sign: float,
    hip_offset: np.ndarray,
    thigh_length: float = THIGH_LENGTH,
    calf_length: float = CALF_LENGTH,
    hip_offset_y: float = HIP_OFFSET_Y,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised FK for one leg across N frames."""
    h = angles[:, 0]
    theta_t = angles[:, 1]
    theta_c = angles[:, 2]
    n_frames = h.shape[0]
    d = hip_offset_y * side_sign

    cos_h, sin_h = np.cos(h), np.sin(h)

    knee_x = -thigh_length * np.sin(theta_t)
    knee_y = np.full(n_frames, d, dtype=np.float64)
    knee_z = -thigh_length * np.cos(theta_t)

    calf_abs = theta_t + theta_c
    foot_x = knee_x - calf_length * np.sin(calf_abs)
    foot_y = knee_y.copy()
    foot_z = knee_z - calf_length * np.cos(calf_abs)

    def _rx_batch(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                vx,
                cos_h * vy - sin_h * vz,
                sin_h * vy + cos_h * vz,
            ],
            axis=-1,
        )

    hip_body = np.broadcast_to(np.asarray(hip_offset, dtype=np.float64), (n_frames, 3)).copy()
    knee_body = np.asarray(hip_offset, dtype=np.float64) + _rx_batch(knee_x, knee_y, knee_z)
    foot_body = np.asarray(hip_offset, dtype=np.float64) + _rx_batch(foot_x, foot_y, foot_z)
    return hip_body, knee_body, foot_body


def batch_inverse_kinematics(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    foot_targets: np.ndarray,
) -> np.ndarray:
    """Batch IK for all 4 legs over N frames."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat = np.asarray(root_quat, dtype=np.float64)
    foot_targets = np.asarray(foot_targets, dtype=np.float64)
    n_frames = root_pos.shape[0]
    R = _quat_to_rot_batch(root_quat)

    foot_body = np.einsum("nij,nmj->nmi", R, foot_targets - root_pos[:, None, :])

    joints = np.empty((n_frames, 12), dtype=np.float64)
    for leg_idx in range(4):
        foot_hip = foot_body[:, leg_idx, :] - HIP_OFFSETS_ARRAY[leg_idx]
        leg_name = LEG_ORDER[leg_idx]
        angles = _solve_leg_ik_batch(foot_hip, LEG_SIDE_SIGN[leg_name])
        joints[:, leg_idx * 3 : leg_idx * 3 + 3] = angles

    np.clip(joints, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], out=joints)
    return joints


def batch_forward_kinematics(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_angles: np.ndarray,
) -> np.ndarray:
    """Batch FK for all 4 legs over N frames."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat = np.asarray(root_quat, dtype=np.float64)
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    n_frames = root_pos.shape[0]
    R = _quat_to_rot_batch(root_quat)

    body_links = np.empty((n_frames, 12, 3), dtype=np.float64)
    for leg_idx in range(4):
        angles = joint_angles[:, leg_idx * 3 : leg_idx * 3 + 3]
        leg_name = LEG_ORDER[leg_idx]
        hip_b, knee_b, foot_b = _forward_kinematics_leg_batch(
            angles,
            LEG_SIDE_SIGN[leg_name],
            HIP_OFFSETS_ARRAY[leg_idx],
        )
        body_links[:, leg_idx, :] = hip_b
        body_links[:, 4 + leg_idx, :] = knee_b
        body_links[:, 8 + leg_idx, :] = foot_b

    return np.einsum("nij,nmj->nmi", R, body_links) + root_pos[:, None, :]


def batch_body_pos_root_relative(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    body_pos_w: np.ndarray,
) -> np.ndarray:
    """Transform world link positions to root-relative frame."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_quat = np.asarray(root_quat, dtype=np.float64)
    body_pos_w = np.asarray(body_pos_w, dtype=np.float64)
    R = _quat_to_rot_batch(root_quat)
    delta = body_pos_w - root_pos[:, None, :]
    return np.einsum("nji,nmj->nmi", R, delta)


__all__ = [
    "JOINT_LIMITS",
    "batch_body_pos_root_relative",
    "batch_forward_kinematics",
    "batch_inverse_kinematics",
    "forward_kinematics_leg",
    "solve_leg_ik",
]
