"""Batched terrain roll, pitch, and height estimation with EMA filtering."""

from __future__ import annotations

import torch
from torch import Tensor

from ..convention import yaw_rotation_matrix_batch
from .types import LEG_ORDER

_EPS = 1e-3
_FL = LEG_ORDER.index("FL")
_FR = LEG_ORDER.index("FR")
_RL = LEG_ORDER.index("RL")
_RR = LEG_ORDER.index("RR")


def _resolve_input_device(*values) -> torch.device:
    devices = [value.device for value in values if isinstance(value, Tensor)]
    if not devices:
        return torch.device("cpu")

    first = devices[0]
    for device in devices[1:]:
        if device != first:
            raise ValueError("batched terrain estimator does not accept tensor inputs on multiple devices")
    return first


def _coerce_tensor(value, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64, device=device)


def _optional_state(value, *, name: str, batch_size: int, device: torch.device, default: Tensor) -> Tensor:
    if value is None:
        return default
    tensor = _coerce_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.expand(batch_size)
    if tensor.ndim == 1 and tensor.shape[0] == batch_size:
        return tensor
    raise ValueError(f"{name} must be a scalar or shape (N,); got {tuple(tensor.shape)}")


def batched_support_roll_pitch(foot_positions) -> tuple[Tensor, Tensor]:
    """Estimate support-plane roll/pitch from world-frame foot positions.

    This mirrors ``raw/scripts/go2fp/base_solver.solve_base_roll_pitch`` but
    vectorized across batch.  Each env solves ``z ~= a*x + b*y + c`` over the
    four support points and converts the plane slopes into roll/pitch.
    """

    device = _resolve_input_device(foot_positions)
    foot_positions_t = _coerce_tensor(foot_positions, device=device)
    if foot_positions_t.ndim != 3 or foot_positions_t.shape[-2:] != (4, 3):
        raise ValueError(f"foot_positions must have shape (N, 4, 3); got {tuple(foot_positions_t.shape)}")

    design = torch.cat(
        [
            foot_positions_t[..., :2],
            torch.ones((*foot_positions_t.shape[:-1], 1), dtype=torch.float64, device=device),
        ],
        dim=-1,
    )
    heights = foot_positions_t[..., 2:3]
    coeffs = torch.linalg.lstsq(design, heights).solution[..., 0]
    slope_x = coeffs[:, 0]
    slope_y = coeffs[:, 1]
    pitch = -torch.atan(slope_x)
    roll = torch.atan(slope_y)
    return roll, pitch


def batched_estimate_terrain(
    foot_positions,
    base_positions,
    base_yaw,
    alpha: float = 0.05,
    initial_roll=None,
    initial_pitch=None,
    initial_height=None,
    contact_state=None,
) -> tuple[Tensor, Tensor, Tensor]:
    device = _resolve_input_device(foot_positions, base_positions, base_yaw, initial_roll, initial_pitch, initial_height, contact_state)
    foot_positions_t = _coerce_tensor(foot_positions, device=device)
    base_positions_t = _coerce_tensor(base_positions, device=device)
    base_yaw_t = _coerce_tensor(base_yaw, device=device)

    if foot_positions_t.ndim != 4 or foot_positions_t.shape[-2:] != (4, 3):
        raise ValueError(f"foot_positions must have shape (N, T, 4, 3); got {tuple(foot_positions_t.shape)}")
    if base_positions_t.ndim != 3 or base_positions_t.shape[-1] != 3:
        raise ValueError(f"base_positions must have shape (N, T, 3); got {tuple(base_positions_t.shape)}")
    if base_yaw_t.ndim != 2:
        raise ValueError(f"base_yaw must have shape (N, T); got {tuple(base_yaw_t.shape)}")

    batch_size, num_frames = int(foot_positions_t.shape[0]), int(foot_positions_t.shape[1])
    if base_positions_t.shape[:2] != (batch_size, num_frames) or base_yaw_t.shape != (batch_size, num_frames):
        raise ValueError("foot_positions, base_positions, and base_yaw must share the same (N, T) dimensions")

    if num_frames == 0:
        empty = torch.empty((batch_size, 0), dtype=torch.float64, device=device)
        return empty, empty.clone(), empty.clone()

    rel = foot_positions_t - base_positions_t[:, :, None, :]
    yaw_rot = yaw_rotation_matrix_batch(base_yaw_t.reshape(-1)).to(dtype=torch.float64).reshape(batch_size, num_frames, 3, 3)
    foot_h = torch.einsum("ntij,ntkj->ntki", yaw_rot, rel)
    contact_state_t = None
    if contact_state is not None:
        contact_state_t = _coerce_tensor(contact_state, device=device)
        if contact_state_t.shape != (batch_size, num_frames, 4):
            raise ValueError(
                "contact_state must have shape "
                f"(N, T, 4); got {tuple(contact_state_t.shape)}"
            )

    if contact_state_t is None:
        left_diff = foot_h[:, :, _FL] - foot_h[:, :, _RL]
        right_diff = foot_h[:, :, _FR] - foot_h[:, :, _RR]
        front_diff = foot_h[:, :, _FL] - foot_h[:, :, _FR]
        back_diff = foot_h[:, :, _RL] - foot_h[:, :, _RR]

        lx, lz = left_diff[..., 0], left_diff[..., 2]
        rx, rz = right_diff[..., 0], right_diff[..., 2]
        fy, fz = front_diff[..., 1], front_diff[..., 2]
        by, bz = back_diff[..., 1], back_diff[..., 2]

        pitch_raw = 0.5 * (torch.atan2(torch.abs(lz), torch.abs(lx) + _EPS) + torch.atan2(torch.abs(rz), torch.abs(rx) + _EPS))
        roll_raw = 0.5 * (torch.atan2(torch.abs(fz), torch.abs(fy) + _EPS) + torch.atan2(torch.abs(bz), torch.abs(by) + _EPS))

        lr_z_mean = 0.5 * (lz + rz)
        fb_z_mean = 0.5 * (fz + bz)
        pitch_signed = torch.where(lr_z_mean > 0.0, -pitch_raw, pitch_raw)
        roll_signed = torch.where(fb_z_mean < 0.0, -roll_raw, roll_raw)
        mean_foot_z = foot_positions_t[..., 2].mean(dim=-1)
    else:
        contact_mask = contact_state_t > 0.5
        front_selector = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64, device=device).view(1, 1, 4)
        rear_selector = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64, device=device).view(1, 1, 4)
        left_selector = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64, device=device).view(1, 1, 4)
        right_selector = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64, device=device).view(1, 1, 4)

        def _group_mean(selector: Tensor) -> tuple[Tensor, Tensor]:
            mask = contact_mask.to(torch.float64) * selector
            count = mask.sum(dim=-1, keepdim=True)
            summed = (foot_positions_t * mask.unsqueeze(-1)).sum(dim=-2)
            mean = torch.where(count > 0.0, summed / count.clamp_min(1.0), torch.zeros_like(summed))
            return mean, count.squeeze(-1) > 0.0

        front_mean, front_valid = _group_mean(front_selector)
        rear_mean, rear_valid = _group_mean(rear_selector)
        left_mean, left_valid = _group_mean(left_selector)
        right_mean, right_valid = _group_mean(right_selector)

        pitch_signed = torch.zeros((batch_size, num_frames), dtype=torch.float64, device=device)
        roll_signed = torch.zeros((batch_size, num_frames), dtype=torch.float64, device=device)

        pitch_mask = front_valid & rear_valid
        roll_mask = left_valid & right_valid

        pitch_delta = front_mean - rear_mean
        roll_delta = left_mean - right_mean
        pitch_signed = torch.where(
            pitch_mask,
            -torch.atan2(pitch_delta[..., 2], pitch_delta[..., 0].abs().clamp_min(_EPS)),
            pitch_signed,
        )
        roll_signed = torch.where(
            roll_mask,
            torch.atan2(roll_delta[..., 2], roll_delta[..., 1].abs().clamp_min(_EPS)),
            roll_signed,
        )

        support_count = contact_mask.sum(dim=-1)
        support_x_max = torch.where(contact_mask, foot_positions_t[..., 0], torch.full_like(foot_positions_t[..., 0], float("-inf")))
        support_x_min = torch.where(contact_mask, foot_positions_t[..., 0], torch.full_like(foot_positions_t[..., 0], float("inf")))
        support_y_max = torch.where(contact_mask, foot_positions_t[..., 1], torch.full_like(foot_positions_t[..., 1], float("-inf")))
        support_y_min = torch.where(contact_mask, foot_positions_t[..., 1], torch.full_like(foot_positions_t[..., 1], float("inf")))
        x_span = torch.where(
            support_count > 0,
            support_x_max.amax(dim=-1) - support_x_min.amin(dim=-1),
            torch.zeros_like(support_count, dtype=torch.float64),
        )
        y_span = torch.where(
            support_count > 0,
            support_y_max.amax(dim=-1) - support_y_min.amin(dim=-1),
            torch.zeros_like(support_count, dtype=torch.float64),
        )
        diagonal_support = support_count == 2
        roll_signed = torch.where(diagonal_support & (x_span > 1.5 * y_span), torch.zeros_like(roll_signed), roll_signed)
        pitch_signed = torch.where(diagonal_support & (y_span > 1.5 * x_span), torch.zeros_like(pitch_signed), pitch_signed)

        support_weight = contact_mask.to(torch.float64)
        support_sum = support_weight.sum(dim=-1)
        support_z = (foot_positions_t[..., 2] * support_weight).sum(dim=-1)
        mean_foot_z = torch.where(
            support_sum > 0.0,
            support_z / support_sum.clamp_min(1.0),
            foot_positions_t[..., 2].mean(dim=-1),
        )

    roll_prev = _optional_state(
        initial_roll,
        name="initial_roll",
        batch_size=batch_size,
        device=device,
        default=torch.zeros(batch_size, dtype=torch.float64, device=device),
    )
    pitch_prev = _optional_state(
        initial_pitch,
        name="initial_pitch",
        batch_size=batch_size,
        device=device,
        default=torch.zeros(batch_size, dtype=torch.float64, device=device),
    )
    height_prev = _optional_state(
        initial_height,
        name="initial_height",
        batch_size=batch_size,
        device=device,
        default=mean_foot_z[:, 0],
    )

    roll_out = torch.empty((batch_size, num_frames), dtype=torch.float64, device=device)
    pitch_out = torch.empty((batch_size, num_frames), dtype=torch.float64, device=device)
    height_out = torch.empty((batch_size, num_frames), dtype=torch.float64, device=device)

    alpha_t = torch.tensor(float(alpha), dtype=torch.float64, device=device)
    one_minus_alpha = 1.0 - alpha_t
    for t in range(num_frames):
        roll_prev = one_minus_alpha * roll_prev + alpha_t * roll_signed[:, t]
        pitch_prev = one_minus_alpha * pitch_prev + alpha_t * pitch_signed[:, t]
        roll_out[:, t] = roll_prev
        pitch_out[:, t] = pitch_prev

        height_prev = 0.8 * mean_foot_z[:, t] + 0.2 * height_prev
        height_out[:, t] = height_prev

    return roll_out, pitch_out, height_out


__all__ = ["batched_estimate_terrain", "batched_support_roll_pitch"]
