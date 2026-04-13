"""Batched gait helpers for the GPU planner."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

GAIT_PARAMS = {
    "trot": {
        "step_freq": 2.0,
        "duty_factor": 0.55,
        "offsets": np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64),
    },
    "walk": {
        "step_freq": 1.0,
        "duty_factor": 0.75,
        "offsets": np.array([0.0, 0.5, 0.75, 0.25], dtype=np.float64),
    },
    "crawl": {
        "step_freq": 0.5,
        "duty_factor": 0.80,
        "offsets": np.array([0.0, 0.25, 0.75, 0.5], dtype=np.float64),
    },
    "pace": {
        "step_freq": 2.0,
        "duty_factor": 0.55,
        "offsets": np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float64),
    },
    "gallop": {
        "step_freq": 3.5,
        "duty_factor": 0.30,
        "offsets": np.array([0.0, 0.05, 0.4, 0.35], dtype=np.float64),
    },
}


def _as_batch_vector(value, *, name: str, device: torch.device | None = None) -> Tensor:
    device = _resolve_input_device(value) if device is None else device
    tensor = _coerce_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor[:, 0]
    raise ValueError(f"{name} must be a scalar, (N,), or (N, 1); got {tuple(tensor.shape)}")


def _as_batch_phase_offsets(phase_offsets, *, device: torch.device | None = None) -> Tensor:
    offsets = _coerce_tensor(phase_offsets, device=_resolve_input_device(phase_offsets) if device is None else device)
    if offsets.ndim == 1:
        if offsets.shape[0] != 4:
            raise ValueError("phase_offsets must have length 4")
        return offsets.reshape(1, 4)
    if offsets.ndim == 2 and offsets.shape[-1] == 4:
        return offsets
    raise ValueError(f"phase_offsets must have shape (4,) or (N, 4); got {tuple(offsets.shape)}")


def _as_batch_contact_seq(contact_seq, *, device: torch.device | None = None) -> Tensor:
    contact = _coerce_contact_tensor(contact_seq, device=_resolve_input_device(contact_seq) if device is None else device)
    if contact.ndim == 2 and contact.shape[-1] == 4:
        return contact.unsqueeze(0)
    if contact.ndim == 3 and contact.shape[-1] == 4:
        return contact
    raise ValueError(f"contact_seq must have shape (N, T, 4) or (T, 4); got {tuple(contact.shape)}")


def _broadcast_batch_size(*sizes: int) -> int:
    batch_size = 1
    for size in sizes:
        if size == 1:
            continue
        if batch_size in (1, size):
            batch_size = size
            continue
        raise ValueError("batched inputs must share a common leading dimension")
    return batch_size


def _resolve_input_device(*values) -> torch.device:
    """Return the shared tensor device, or CPU when no tensor inputs are present."""
    devices = []
    for value in values:
        if isinstance(value, Tensor):
            devices.append(value.device)
    if not devices:
        return torch.device("cpu")

    first = devices[0]
    for device in devices[1:]:
        if device != first:
            raise ValueError("batched gait helpers do not accept tensor inputs on multiple devices")
    return first


def _coerce_tensor(value, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64, device=device)


def _coerce_contact_tensor(value, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _pack_event_frames(mask: Tensor, *, sentinel: int) -> tuple[Tensor, Tensor]:
    """Compact event masks into ordered frame indices without sorting."""
    batch_size, n_legs, width = mask.shape
    values = torch.full((batch_size, n_legs, width), sentinel, dtype=torch.int64, device=mask.device)
    if width == 0:
        return values, torch.empty((batch_size, n_legs, 0), dtype=torch.bool, device=mask.device)

    event_ranks = torch.cumsum(mask.to(torch.int64), dim=-1) - 1
    event_positions = mask.nonzero(as_tuple=True)
    if event_positions[0].numel() > 0:
        event_frames = torch.arange(1, width + 1, dtype=torch.int64, device=mask.device).view(1, 1, -1)
        expanded_frames = event_frames.expand(batch_size, n_legs, width)
        values[event_positions[0], event_positions[1], event_ranks[event_positions]] = expanded_frames[event_positions]
    return values, values != sentinel


def batched_gait_schedule(
    t0,
    n_frames: int,
    dt: float,
    step_freq,
    duty_factor,
    phase_offsets,
) -> Tensor:
    """Generate batched contact schedules with shape (N, n_frames, 4).

    Scalar inputs broadcast to a single batch item. All tensor inputs must share
    one device; mixed-device inputs raise ``ValueError``. When no tensor inputs
    are present, CPU is used.
    """
    device = _resolve_input_device(t0, step_freq, duty_factor, phase_offsets)
    t0_b = _as_batch_vector(t0, name="t0", device=device)
    step_freq_b = _as_batch_vector(step_freq, name="step_freq", device=device)
    duty_factor_b = _as_batch_vector(duty_factor, name="duty_factor", device=device)
    phase_offsets_b = _as_batch_phase_offsets(phase_offsets, device=device)

    batch_size = _broadcast_batch_size(
        int(t0_b.shape[0]),
        int(step_freq_b.shape[0]),
        int(duty_factor_b.shape[0]),
        int(phase_offsets_b.shape[0]),
    )

    if t0_b.shape[0] == 1:
        t0_b = t0_b.expand(batch_size)
    elif t0_b.shape[0] != batch_size:
        raise ValueError("t0 batch size must match other batched gait inputs")

    if step_freq_b.shape[0] == 1:
        step_freq_b = step_freq_b.expand(batch_size)
    elif step_freq_b.shape[0] != batch_size:
        raise ValueError("step_freq batch size must match other batched gait inputs")

    if duty_factor_b.shape[0] == 1:
        duty_factor_b = duty_factor_b.expand(batch_size)
    elif duty_factor_b.shape[0] != batch_size:
        raise ValueError("duty_factor batch size must match other batched gait inputs")

    if phase_offsets_b.shape[0] == 1:
        phase_offsets_b = phase_offsets_b.expand(batch_size, -1)
    elif phase_offsets_b.shape[0] != batch_size:
        raise ValueError("phase_offsets batch size must match other batched gait inputs")

    frame_index = torch.arange(int(n_frames), dtype=torch.float64, device=t0_b.device)
    t = t0_b[:, None] + frame_index[None, :] * float(dt)
    phase = torch.remainder(t[:, :, None] * step_freq_b[:, None, None] + phase_offsets_b[:, None, :], 1.0)
    return (phase < duty_factor_b[:, None, None]).to(torch.float32)


def batched_next_touchdown_times(step_freq, phase_offsets) -> Tensor:
    """Return time-to-next-touchdown with shape (N, 4).

    Scalar inputs broadcast to a single batch item. Mixed-device tensor inputs
    raise ``ValueError``.
    """
    device = _resolve_input_device(step_freq, phase_offsets)
    step_freq_b = _as_batch_vector(step_freq, name="step_freq", device=device)
    phase_offsets_b = _as_batch_phase_offsets(phase_offsets, device=device)

    batch_size = _broadcast_batch_size(int(step_freq_b.shape[0]), int(phase_offsets_b.shape[0]))

    if step_freq_b.shape[0] == 1:
        step_freq_b = step_freq_b.expand(batch_size)
    elif step_freq_b.shape[0] != batch_size:
        raise ValueError("step_freq batch size must match phase_offsets batch size")

    if phase_offsets_b.shape[0] == 1:
        phase_offsets_b = phase_offsets_b.expand(batch_size, -1)
    elif phase_offsets_b.shape[0] != batch_size:
        raise ValueError("phase_offsets batch size must match step_freq batch size")

    cycles = torch.remainder(1.0 - phase_offsets_b, 1.0)
    cycles = torch.where(cycles < 1e-9, torch.ones_like(cycles), cycles)
    return cycles / step_freq_b[:, None]


def batched_stance_time(step_freq, duty_factor) -> Tensor:
    """Return stance duration with shape (N,).

    Scalar inputs broadcast to a single batch item.
    Mixed-device tensor inputs raise ``ValueError``.
    """
    device = _resolve_input_device(step_freq, duty_factor)
    step_freq_b = _as_batch_vector(step_freq, name="step_freq", device=device)
    duty_factor_b = _as_batch_vector(duty_factor, name="duty_factor", device=device)

    batch_size = _broadcast_batch_size(int(step_freq_b.shape[0]), int(duty_factor_b.shape[0]))

    if step_freq_b.shape[0] == 1:
        step_freq_b = step_freq_b.expand(batch_size)
    elif step_freq_b.shape[0] != batch_size:
        raise ValueError("step_freq batch size must match duty_factor batch size")

    if duty_factor_b.shape[0] == 1:
        duty_factor_b = duty_factor_b.expand(batch_size)
    elif duty_factor_b.shape[0] != batch_size:
        raise ValueError("duty_factor batch size must match step_freq batch size")

    return duty_factor_b / step_freq_b


def batched_legs_requiring_touchdown(contact_seq) -> Tensor:
    """Return a batched boolean mask for legs that touch down within each segment.

    A 2D ``(T, 4)`` input is treated as a single segment and returns ``(1, 4)``.
    """
    contact = _as_batch_contact_seq(contact_seq, device=_resolve_input_device(contact_seq))
    diff = torch.diff(contact, dim=1)
    return torch.any(diff > 0.5, dim=1)


def batched_detect_swing_events(contact_seq):
    """Detect swing events for each batch element.

    Returns a dictionary of fixed-shape tensors with shape ``(N, 4, T - 1)``:
    ``lift_off`` and ``touch_down`` store ordered event frame indices with
    padding set to ``T`` - one past the last valid frame index - for absent
    events, and ``lift_off_valid`` / ``touch_down_valid`` are boolean masks
    marking which entries are real events.

    A 2D ``(T, 4)`` input is promoted to a batch of size 1 and keeps the
    leading batch dimension.
    """
    contact = _as_batch_contact_seq(contact_seq, device=_resolve_input_device(contact_seq))
    contact_bool = contact if contact.dtype == torch.bool else contact > 0.5
    event_width = int(max(contact.shape[1] - 1, 0))
    sentinel = int(contact.shape[1])
    if event_width == 0:
        empty = torch.empty((contact.shape[0], 4, 0), dtype=torch.int64, device=contact.device)
        empty_valid = torch.empty((contact.shape[0], 4, 0), dtype=torch.bool, device=contact.device)
        return {
            "lift_off": empty,
            "lift_off_valid": empty_valid,
            "touch_down": empty.clone(),
            "touch_down_valid": empty_valid.clone(),
        }

    prev_contact = contact_bool[:, :-1, :].permute(0, 2, 1)
    next_contact = contact_bool[:, 1:, :].permute(0, 2, 1)
    lift_off, lift_off_valid = _pack_event_frames(prev_contact & ~next_contact, sentinel=sentinel)
    touch_down, touch_down_valid = _pack_event_frames(~prev_contact & next_contact, sentinel=sentinel)

    return {
        "lift_off": lift_off,
        "lift_off_valid": lift_off_valid,
        "touch_down": touch_down,
        "touch_down_valid": touch_down_valid,
    }


__all__ = [
    "GAIT_PARAMS",
    "batched_detect_swing_events",
    "batched_gait_schedule",
    "batched_legs_requiring_touchdown",
    "batched_next_touchdown_times",
    "batched_stance_time",
]
