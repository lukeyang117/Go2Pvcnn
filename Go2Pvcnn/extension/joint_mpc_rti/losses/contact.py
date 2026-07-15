"""Stance locking, surface contact, and touchdown residuals."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.losses.barriers import localized_relaxed_barrier, masked_mean, relaxed_barrier


def stance_losses(foot_pos_w: Tensor, queried_height_w: Tensor, contact_state: Tensor, *, dt: float) -> dict[str, Tensor]:
    foot = torch.as_tensor(foot_pos_w)
    height = torch.as_tensor(queried_height_w, dtype=foot.dtype, device=foot.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    if height.shape != foot[..., 2].shape or contact.shape != height.shape:
        raise ValueError("queried_height_w and contact_state must match foot [B,T,4]")
    ground_error = foot[..., 2] - height
    ground = masked_mean(ground_error * ground_error, contact, dims=(1, 2))
    consecutive_contact = torch.logical_and(contact[:, 1:], contact[:, :-1])
    xy_step = foot[:, 1:, :, :2] - foot[:, :-1, :, :2]
    xy_step_sq = (xy_step * xy_step).sum(dim=-1)
    xy_lock = masked_mean(xy_step_sq, consecutive_contact, dims=(1, 2))
    slip = masked_mean(xy_step_sq / (float(dt) ** 2), consecutive_contact, dims=(1, 2))
    return {
        "stance_xy_lock": xy_lock,
        "stance_ground_contact": ground,
        "stance_slip_velocity": slip,
    }


def touchdown_losses(
    *,
    touchdown_pos_w: Tensor,
    queried_height_w: Tensor,
    queried_valid: Tensor,
) -> dict[str, Tensor]:
    touchdown = torch.as_tensor(touchdown_pos_w)
    height = torch.as_tensor(queried_height_w, dtype=touchdown.dtype, device=touchdown.device)
    valid = torch.as_tensor(queried_valid, dtype=torch.bool, device=touchdown.device)
    if height.shape != touchdown[..., 2].shape or valid.shape != height.shape:
        raise ValueError("queried touchdown fields must match touchdown [B,4]")
    height_error = touchdown[..., 2] - height
    return {
        "touchdown_ground_height": (height_error * height_error).mean(dim=-1),
        "touchdown_valid_map": torch.logical_not(valid).to(dtype=touchdown.dtype).mean(dim=-1),
    }


def swing_losses(
    *,
    foot_pos_w: Tensor,
    nominal_foot_pos_w: Tensor,
    queried_height_w: Tensor,
    swing_mask: Tensor,
    swing_weight: Tensor | None = None,
    dt: float,
    terrain_margin: float = 0.02,
    barrier_relaxation: float = 0.01,
) -> dict[str, Tensor]:
    foot = torch.as_tensor(foot_pos_w)
    nominal = torch.as_tensor(nominal_foot_pos_w, dtype=foot.dtype, device=foot.device)
    height = torch.as_tensor(queried_height_w, dtype=foot.dtype, device=foot.device)
    swing = torch.as_tensor(swing_mask, dtype=torch.bool, device=foot.device)
    weight = swing.to(foot.dtype) if swing_weight is None else torch.as_tensor(
        swing_weight, dtype=foot.dtype, device=foot.device
    )
    shape_error = ((foot - nominal) ** 2).sum(dim=-1)
    shape_loss = masked_mean(shape_error, weight, dims=(1, 2))
    terrain_margin_value = foot[..., 2] - height - float(terrain_margin)
    clearance = masked_mean(
        localized_relaxed_barrier(
            terrain_margin_value,
            activation_margin=0.005,
            relaxation=barrier_relaxation,
        ),
        weight,
        dims=(1, 2),
    )
    velocity = (foot[:, 1:] - foot[:, :-1]) / float(dt)
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / float(dt)
    acceleration_sq = (acceleration * acceleration).sum(dim=-1)
    acceleration_mask = torch.logical_and(swing[:, 2:], torch.logical_and(swing[:, 1:-1], swing[:, :-2]))
    velocity_smoothness = masked_mean(acceleration_sq, acceleration_mask, dims=(1, 2))
    touchdown_velocity = (velocity[:, -1] ** 2).sum(dim=-1).mean(dim=1)
    return {
        "swing_nominal_shape": shape_loss,
        "terrain_swing_clearance": clearance,
        "swing_velocity_smoothness": velocity_smoothness,
        "touchdown_velocity": touchdown_velocity,
    }


def touchdown_geometry_losses(
    touchdown_pos_root: Tensor,
    *,
    min_reach: float,
    max_reach: float,
    min_left_right_separation: float,
    barrier_relaxation: float = 0.01,
) -> dict[str, Tensor]:
    touchdown = torch.as_tensor(touchdown_pos_root)
    if touchdown.ndim != 3 or tuple(touchdown.shape[-2:]) != (4, 3):
        raise ValueError("touchdown_pos_root must have shape [B,4,3]")
    reach = torch.linalg.vector_norm(touchdown, dim=-1)
    reach_cost = (
        relaxed_barrier(reach - float(min_reach), relaxation=barrier_relaxation)
        + relaxed_barrier(float(max_reach) - reach, relaxation=barrier_relaxation)
    ).mean(dim=1)
    front_separation = torch.abs(touchdown[:, 0, 1] - touchdown[:, 1, 1])
    rear_separation = torch.abs(touchdown[:, 2, 1] - touchdown[:, 3, 1])
    separation = torch.stack((front_separation, rear_separation), dim=-1)
    separation_cost = relaxed_barrier(
        separation - float(min_left_right_separation), relaxation=barrier_relaxation
    ).mean(dim=1)
    return {
        "touchdown_reach_margin": reach_cost,
        "touchdown_foot_separation": separation_cost,
    }


__all__ = ["stance_losses", "swing_losses", "touchdown_geometry_losses", "touchdown_losses"]
