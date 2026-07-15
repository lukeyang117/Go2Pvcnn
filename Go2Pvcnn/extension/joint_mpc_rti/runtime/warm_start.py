"""Shift and measured-state injection for rolling RTI warm starts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class JointMpcWarmStart:
    state: Tensor
    control: Tensor
    dual: Tensor | None = None


def shift_warm_start(
    previous_state: Tensor,
    previous_control: Tensor,
    measured_state: Tensor,
    previous_dual: Tensor | None = None,
) -> JointMpcWarmStart:
    """Shift one horizon interval and inject the measured state at node zero."""
    state = torch.as_tensor(previous_state)
    control = torch.as_tensor(previous_control, dtype=state.dtype, device=state.device)
    measured = torch.as_tensor(measured_state, dtype=state.dtype, device=state.device)
    if state.ndim != 3 or int(state.shape[-1]) != 18:
        raise ValueError("previous_state must have shape [B,H+1,18]")
    if control.ndim != 3 or tuple(control.shape) != (int(state.shape[0]), int(state.shape[1]) - 1, 18):
        raise ValueError("previous_control must have shape [B,H,18]")
    if measured.shape != state[:, 0].shape:
        raise ValueError("measured_state must have shape [B,18]")
    terminal_state = state[:, -1] + (state[:, -1] - state[:, -2])
    shifted_state = torch.cat((measured.unsqueeze(1), state[:, 2:], terminal_state.unsqueeze(1)), dim=1)
    shifted_control = torch.cat((control[:, 1:], control[:, -1:]), dim=1)
    shifted_dual = None
    if previous_dual is not None:
        dual = torch.as_tensor(previous_dual, dtype=state.dtype, device=state.device)
        if dual.ndim < 2 or int(dual.shape[0]) != int(state.shape[0]):
            raise ValueError("previous_dual must be batched")
        shifted_dual = torch.cat((dual[:, 1:], dual[:, -1:]), dim=1)
    return JointMpcWarmStart(state=shifted_state, control=shifted_control, dual=shifted_dual)


__all__ = ["JointMpcWarmStart", "shift_warm_start"]
