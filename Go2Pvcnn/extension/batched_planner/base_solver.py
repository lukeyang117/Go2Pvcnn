"""Batched base trajectory solver for the GPU planner."""

from __future__ import annotations

import torch
from torch import Tensor

from ..convention import euler_to_quat_batch
from .types import HIP_HEIGHT

BODY_COLLISION_SAMPLES = torch.tensor(
    [
        [0.1881, 0.04675, -0.057],
        [0.1881, -0.04675, -0.057],
        [-0.1881, 0.04675, -0.057],
        [-0.1881, -0.04675, -0.057],
        [0.293, 0.0, -0.107],
        [0.14, 0.0, -0.057],
        [0.0, 0.0, -0.057],
        [-0.14, 0.0, -0.057],
    ],
    dtype=torch.float64,
)


def _resolve_input_device(*values, context: str = "batched base solver") -> torch.device:
    devices = [value.device for value in values if isinstance(value, Tensor)]
    if not devices:
        return torch.device("cpu")
    first = devices[0]
    if any(device != first for device in devices[1:]):
        device_list = ", ".join(dict.fromkeys(str(device) for device in devices))
        raise ValueError(f"{context} requires all tensor inputs to live on one device; mixed-device inputs: {device_list}")
    return first


def _terrain_device(terrain) -> torch.device | None:
    heightmaps = getattr(terrain, "heightmaps", None)
    if isinstance(heightmaps, Tensor):
        return heightmaps.device
    device = getattr(terrain, "device", None)
    if device is None:
        return None
    return torch.device(device)


def _require_terrain_device(terrain, device: torch.device, *, context: str = "batched base solver") -> None:
    terrain_device = _terrain_device(terrain)
    if terrain_device is not None and terrain_device != device:
        raise ValueError(
            f"{context} requires terrain and tensor inputs on the same device; "
            f"got terrain on {terrain_device} and tensor inputs on {device}"
        )


def _coerce_tensor(value, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64, device=device)


def _as_shape(name: str, value, *, device: torch.device, shape: tuple[int, ...]) -> Tensor:
    tensor = _coerce_tensor(value, device=device)
    if tensor.ndim != len(shape) + 1 or tuple(tensor.shape[1:]) != shape:
        raise ValueError(f"{name} must have shape (N, {', '.join(str(dim) for dim in shape)}); got {tuple(tensor.shape)}")
    return tensor


def _as_batch_vector(name: str, value, *, device: torch.device) -> Tensor:
    tensor = _coerce_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    raise ValueError(f"{name} must be a scalar or shape (N,); got {tuple(tensor.shape)}")


def _quat_to_rot_batch(quat: Tensor) -> Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=-1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def batched_integrate_base_planar(initial_pos_xy, initial_yaw, vx, vy, yaw_rate, n_frames: int, dt: float) -> tuple[Tensor, Tensor]:
    device = _resolve_input_device(initial_pos_xy, initial_yaw, vx, vy, yaw_rate)
    initial_pos_xy_t = _as_shape("initial_pos_xy", initial_pos_xy, device=device, shape=(2,))
    initial_yaw_t = _as_batch_vector("initial_yaw", initial_yaw, device=device)
    vx_t = _as_batch_vector("vx", vx, device=device)
    vy_t = _as_batch_vector("vy", vy, device=device)
    yaw_rate_t = _as_batch_vector("yaw_rate", yaw_rate, device=device)

    batch_size = int(initial_pos_xy_t.shape[0])
    if not all(t.shape[0] == batch_size for t in (initial_yaw_t, vx_t, vy_t, yaw_rate_t)):
        raise ValueError("initial_pos_xy, initial_yaw, vx, vy, and yaw_rate must share batch size")

    if n_frames <= 0:
        return (
            torch.empty((batch_size, 0, 2), dtype=torch.float64, device=device),
            torch.empty((batch_size, 0), dtype=torch.float64, device=device),
        )

    frame_idx = torch.arange(n_frames, dtype=torch.float64, device=device)
    yaw = initial_yaw_t[:, None] + frame_idx[None, :] * yaw_rate_t[:, None] * float(dt)
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    dx = vx_t[:, None] * c - vy_t[:, None] * s
    dy = vx_t[:, None] * s + vy_t[:, None] * c
    delta_xy = torch.stack([dx, dy], dim=-1)
    pos_xy = initial_pos_xy_t[:, None, :] + torch.cumsum(delta_xy * float(dt), dim=1)
    return pos_xy, yaw


def batched_solve_base_height(terrain_height, foot_targets, contact_seq, hip_height: float = HIP_HEIGHT, smooth_factor: float = 0.3) -> Tensor:
    device = _resolve_input_device(terrain_height, foot_targets, contact_seq)
    terrain_height_t = _as_batch_vector("terrain_height", terrain_height, device=device) if isinstance(terrain_height, Tensor) and terrain_height.ndim == 1 else _as_shape("terrain_height", terrain_height, device=device, shape=(contact_seq.shape[1] if isinstance(contact_seq, Tensor) and contact_seq.ndim == 3 else _coerce_tensor(contact_seq, device=device).shape[1],))
    foot_targets_t = _as_shape("foot_targets", foot_targets, device=device, shape=(terrain_height_t.shape[1], 4, 3))
    contact_seq_t = _as_shape("contact_seq", contact_seq, device=device, shape=(terrain_height_t.shape[1], 4))

    foot_z = foot_targets_t[..., 2]
    weighted_sum = (contact_seq_t * foot_z).sum(dim=-1)
    contact_sum = contact_seq_t.sum(dim=-1)
    support_z = torch.where(contact_sum > 1e-9, weighted_sum / contact_sum, terrain_height_t)
    target_z = support_z + float(hip_height)

    batch_size, num_frames = target_z.shape
    base_z = torch.empty((batch_size, num_frames), dtype=torch.float64, device=device)
    prev = target_z[:, 0]
    sf = float(max(0.0, min(1.0, smooth_factor)))
    for t in range(num_frames):
        prev = (1.0 - sf) * prev + sf * target_z[:, t]
        base_z[:, t] = prev
    return base_z


def batched_solve_base_orientation(terrain_roll, terrain_pitch, yaw, max_roll: float = 0.35, max_pitch: float = 0.45) -> Tensor:
    device = _resolve_input_device(terrain_roll, terrain_pitch, yaw)
    terrain_roll_t = _coerce_tensor(terrain_roll, device=device)
    terrain_pitch_t = _coerce_tensor(terrain_pitch, device=device)
    yaw_t = _coerce_tensor(yaw, device=device)
    if terrain_roll_t.shape != terrain_pitch_t.shape or terrain_roll_t.shape != yaw_t.shape:
        raise ValueError("terrain_roll, terrain_pitch, and yaw must have the same shape")
    return euler_to_quat_batch(
        torch.clamp(terrain_roll_t, -max_roll, max_roll),
        torch.clamp(terrain_pitch_t, -max_pitch, max_pitch),
        yaw_t,
    ).to(dtype=torch.float64)


def batched_body_clearance_adjustment(base_pos, base_quat, terrain, body_samples: Tensor = BODY_COLLISION_SAMPLES, margin: float = 0.012) -> Tensor:
    device = _resolve_input_device(base_pos, base_quat, body_samples)
    _require_terrain_device(terrain, device)
    base_pos_t = _as_shape("base_pos", base_pos, device=device, shape=(base_quat.shape[1] if isinstance(base_quat, Tensor) and base_quat.ndim == 3 else _coerce_tensor(base_quat, device=device).shape[1], 3))
    base_quat_t = _as_shape("base_quat", base_quat, device=device, shape=(base_pos_t.shape[1], 4))
    samples_t = _coerce_tensor(body_samples, device=device)
    if samples_t.ndim != 2 or samples_t.shape[1] != 3:
        raise ValueError(f"body_samples must have shape (S, 3); got {tuple(samples_t.shape)}")

    batch_size, num_frames = base_pos_t.shape[:2]
    rot = _quat_to_rot_batch(base_quat_t.reshape(-1, 4)).reshape(batch_size, num_frames, 3, 3)
    world_samples = base_pos_t[:, :, None, :] + torch.einsum("ntij,sj->ntsi", rot, samples_t)
    sample_xy = world_samples[..., :2].reshape(batch_size, num_frames * samples_t.shape[0], 2)
    terrain_heights = terrain.height_at(sample_xy).reshape(batch_size, num_frames, samples_t.shape[0])
    deficits = terrain_heights + float(margin) - world_samples[..., 2]
    return torch.clamp_min(deficits.max(dim=-1).values, 0.0)


def batched_solve_base_trajectory(
    initial_pos,
    initial_yaw,
    vx,
    vy,
    yaw_rate,
    n_frames: int,
    dt: float,
    terrain,
    foot_targets,
    contact_seq,
    terrain_roll,
    terrain_pitch,
    terrain_height,
    hip_height: float = HIP_HEIGHT,
    body_clearance_margin: float = 0.012,
) -> tuple[Tensor, Tensor]:
    device = _resolve_input_device(initial_pos, initial_yaw, vx, vy, yaw_rate, foot_targets, contact_seq, terrain_roll, terrain_pitch, terrain_height)
    _require_terrain_device(terrain, device)
    initial_pos_t = _as_shape("initial_pos", initial_pos, device=device, shape=(3,))
    initial_yaw_t = _as_batch_vector("initial_yaw", initial_yaw, device=device)
    vx_t = _as_batch_vector("vx", vx, device=device)
    vy_t = _as_batch_vector("vy", vy, device=device)
    yaw_rate_t = _as_batch_vector("yaw_rate", yaw_rate, device=device)
    foot_targets_t = _as_shape("foot_targets", foot_targets, device=device, shape=(n_frames, 4, 3))
    contact_seq_t = _as_shape("contact_seq", contact_seq, device=device, shape=(n_frames, 4))
    terrain_roll_t = _as_shape("terrain_roll", terrain_roll, device=device, shape=(n_frames,))
    terrain_pitch_t = _as_shape("terrain_pitch", terrain_pitch, device=device, shape=(n_frames,))
    terrain_height_t = _as_shape("terrain_height", terrain_height, device=device, shape=(n_frames,))

    batch_size = int(initial_pos_t.shape[0])
    if not all(t.shape[0] == batch_size for t in (initial_yaw_t, vx_t, vy_t, yaw_rate_t, foot_targets_t, contact_seq_t, terrain_roll_t, terrain_pitch_t, terrain_height_t)):
        raise ValueError("all batched base solver inputs must share batch size")

    pos_xy, yaw = batched_integrate_base_planar(initial_pos_t[:, :2], initial_yaw_t, vx_t, vy_t, yaw_rate_t, n_frames, dt)
    z = batched_solve_base_height(terrain_height_t, foot_targets_t, contact_seq_t, hip_height=hip_height)
    quat = batched_solve_base_orientation(terrain_roll_t, terrain_pitch_t, yaw)
    root_pos = torch.cat([pos_xy, z.unsqueeze(-1)], dim=-1)
    z_adjustment = batched_body_clearance_adjustment(root_pos, quat, terrain, margin=body_clearance_margin)
    root_pos = root_pos.clone()
    root_pos[..., 2] = root_pos[..., 2] + z_adjustment
    return root_pos, quat


__all__ = [
    "BODY_COLLISION_SAMPLES",
    "batched_body_clearance_adjustment",
    "batched_integrate_base_planar",
    "batched_solve_base_height",
    "batched_solve_base_orientation",
    "batched_solve_base_trajectory",
]
