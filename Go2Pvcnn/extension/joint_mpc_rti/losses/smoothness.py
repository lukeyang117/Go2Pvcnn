"""Control and acceleration regularization for rolling RTI continuity."""

from __future__ import annotations

import torch
from torch import Tensor


def smoothness_losses(control: Tensor, *, previous_control: Tensor, dt: float) -> dict[str, Tensor]:
    controls = torch.as_tensor(control)
    previous = torch.as_tensor(previous_control, dtype=controls.dtype, device=controls.device)
    if controls.ndim != 3 or int(controls.shape[-1]) != 18:
        raise ValueError("control must have shape [B,H,18]")
    if previous.shape != controls[:, 0].shape:
        raise ValueError("previous_control must have shape [B,18]")
    delta = controls[:, 1:] - controls[:, :-1]
    control_rate = (delta * delta).mean(dim=(1, 2))
    first = controls[:, 0] - previous
    first_continuity = (first * first).mean(dim=1)
    acceleration = delta / float(dt)
    root_acceleration = (acceleration[..., :6] ** 2).mean(dim=(1, 2))
    joint_acceleration = (acceleration[..., 6:] ** 2).mean(dim=(1, 2))
    return {
        "control_rate": control_rate,
        "first_control_continuity": first_continuity,
        "joint_acceleration": joint_acceleration,
        "root_acceleration": root_acceleration,
    }


__all__ = ["smoothness_losses"]
