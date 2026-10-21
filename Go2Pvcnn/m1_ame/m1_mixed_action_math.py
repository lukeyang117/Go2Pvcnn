"""Pure tensor helpers for the M1 mixed leg/wheel action."""

from __future__ import annotations

import torch
from torch import Tensor


def split_m1_mixed_action(
    actions: Tensor,
    leg_indices: tuple[int, ...],
    wheel_indices: tuple[int, ...],
    leg_scale: float,
    wheel_scale: float,
    default_leg_pos: Tensor,
    wheel_velocity_limit: float | None = None,
    leg_position_lower: Tensor | None = None,
    leg_position_upper: Tensor | None = None,
    normalized_action_limit: float | None = None,
    previous_leg_pos_target: Tensor | None = None,
    leg_target_delta_limit: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Convert canonical 16-D actions into leg position and wheel velocity targets."""

    if actions.ndim != 2:
        raise ValueError(f"M1 mixed actions must be rank-2, got shape {tuple(actions.shape)}")
    if default_leg_pos.ndim != 2 or default_leg_pos.shape[0] != actions.shape[0]:
        raise ValueError(
            "default_leg_pos must have shape (num_envs, num_leg_joints), "
            f"got {tuple(default_leg_pos.shape)} for actions {tuple(actions.shape)}"
        )
    if len(leg_indices) != default_leg_pos.shape[1]:
        raise ValueError(
            f"leg index count {len(leg_indices)} does not match default pose width {default_leg_pos.shape[1]}"
        )

    leg_index_tensor = torch.as_tensor(leg_indices, device=actions.device, dtype=torch.long)
    wheel_index_tensor = torch.as_tensor(wheel_indices, device=actions.device, dtype=torch.long)
    if normalized_action_limit is not None:
        limit = abs(float(normalized_action_limit))
        actions = torch.clamp(actions, min=-limit, max=limit)
    leg_action = actions.index_select(-1, leg_index_tensor)
    wheel_action = actions.index_select(-1, wheel_index_tensor)
    position_targets = default_leg_pos + float(leg_scale) * leg_action
    if leg_position_lower is not None or leg_position_upper is not None:
        if leg_position_lower is None or leg_position_upper is None:
            raise ValueError("leg_position_lower and leg_position_upper must be provided together")
        lower = torch.as_tensor(leg_position_lower, device=actions.device, dtype=position_targets.dtype)
        upper = torch.as_tensor(leg_position_upper, device=actions.device, dtype=position_targets.dtype)
        if lower.shape != position_targets.shape and lower.ndim != 1:
            raise ValueError("leg position limits must be [num_envs, num_leg_joints] or [num_leg_joints]")
        if upper.shape != position_targets.shape and upper.ndim != 1:
            raise ValueError("leg position limits must be [num_envs, num_leg_joints] or [num_leg_joints]")
        if lower.shape[-1] != position_targets.shape[-1] or upper.shape[-1] != position_targets.shape[-1]:
            raise ValueError("leg position limits must match the number of leg joints")
        position_targets = torch.maximum(position_targets, lower)
        position_targets = torch.minimum(position_targets, upper)
    if previous_leg_pos_target is not None and leg_target_delta_limit is not None:
        previous = torch.as_tensor(previous_leg_pos_target, device=actions.device, dtype=position_targets.dtype)
        delta = float(abs(leg_target_delta_limit))
        position_targets = torch.maximum(position_targets, previous - delta)
        position_targets = torch.minimum(position_targets, previous + delta)
    velocity_targets = float(wheel_scale) * wheel_action
    if wheel_velocity_limit is not None:
        limit = abs(float(wheel_velocity_limit))
        velocity_targets = torch.clamp(velocity_targets, min=-limit, max=limit)
    return position_targets, velocity_targets


__all__ = ["split_m1_mixed_action"]
