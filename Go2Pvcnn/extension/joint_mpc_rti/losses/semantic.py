"""Continuous semantic clearances without cross/avoid behavior gates."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.losses.barriers import masked_mean, relaxed_barrier


def small_object_losses(
    *,
    foot_pos_w: Tensor,
    foot_small_distance: Tensor,
    small_top_height: Tensor,
    small_distance_touchdown: Tensor,
    link_pos_w: Tensor,
    link_small_distance: Tensor,
    swing_mask: Tensor,
    stance_mask: Tensor,
    extra_margin: float,
    influence_radius: float = 0.08,
    temperature: float = 0.02,
    touchdown_margin: float = 0.02,
    link_margin_xy: float = 0.015,
    link_margin_z: float = 0.01,
    barrier_relaxation: float = 0.01,
) -> dict[str, Tensor]:
    foot = torch.as_tensor(foot_pos_w)
    foot_distance = torch.as_tensor(foot_small_distance, dtype=foot.dtype, device=foot.device)
    top = torch.as_tensor(small_top_height, dtype=foot.dtype, device=foot.device)
    swing = torch.as_tensor(swing_mask, dtype=torch.bool, device=foot.device)
    stance = torch.as_tensor(stance_mask, dtype=torch.bool, device=foot.device)
    influence = torch.sigmoid((float(influence_radius) - foot_distance) / float(temperature))
    over_margin = foot[..., 2] - top - float(extra_margin)
    foot_over_raw = influence * relaxed_barrier(over_margin, relaxation=barrier_relaxation)
    foot_over = masked_mean(foot_over_raw, swing, dims=tuple(range(1, foot_over_raw.ndim)))
    touchdown_distance = torch.as_tensor(small_distance_touchdown, dtype=foot.dtype, device=foot.device)
    touchdown_raw = relaxed_barrier(touchdown_distance - float(touchdown_margin), relaxation=barrier_relaxation)
    touchdown_avoidance = touchdown_raw.reshape(foot.shape[0], -1).mean(dim=1)
    link = torch.as_tensor(link_pos_w, dtype=foot.dtype, device=foot.device)
    link_distance = torch.as_tensor(link_small_distance, dtype=foot.dtype, device=foot.device)
    link_top = torch.broadcast_to(top, link_distance.shape)
    vertical_influence = torch.sigmoid((link_top + float(link_margin_z) - link[..., 2]) / float(temperature))
    link_raw = vertical_influence * relaxed_barrier(
        link_distance - float(link_margin_xy), relaxation=barrier_relaxation
    )
    link_clearance = link_raw.reshape(foot.shape[0], -1).mean(dim=1)
    stance_touchdown = masked_mean(
        relaxed_barrier(foot_distance - float(touchdown_margin), relaxation=barrier_relaxation),
        stance,
        dims=tuple(range(1, foot_distance.ndim)),
    )
    return {
        "small_object_foot_over": foot_over,
        "small_object_touchdown_avoidance": touchdown_avoidance + stance_touchdown,
        "small_object_link_clearance": link_clearance,
    }


def large_obstacle_losses(
    *,
    root_footprint_distance: Tensor,
    body_distance: Tensor,
    foot_distance: Tensor,
    knee_shank_distance: Tensor,
    terminal_distance: Tensor,
    terminal_approach_speed: Tensor,
    root_margin: float = 0.12,
    body_margin: float = 0.08,
    foot_margin: float = 0.03,
    link_margin: float = 0.04,
    terminal_margin: float = 0.16,
    barrier_relaxation: float = 0.01,
) -> dict[str, Tensor]:
    root_distance = torch.as_tensor(root_footprint_distance)
    dtype = root_distance.dtype
    device = root_distance.device

    def barrier_mean(distance: Tensor, margin: float) -> Tensor:
        value = torch.as_tensor(distance, dtype=dtype, device=device)
        return relaxed_barrier(value - float(margin), relaxation=barrier_relaxation).reshape(value.shape[0], -1).mean(dim=1)

    terminal = torch.as_tensor(terminal_distance, dtype=dtype, device=device)
    approach = torch.as_tensor(terminal_approach_speed, dtype=dtype, device=device)
    terminal_risk = relaxed_barrier(terminal - float(terminal_margin), relaxation=barrier_relaxation) + torch.relu(-approach) ** 2
    return {
        "large_root_footprint_barrier": barrier_mean(root_distance, root_margin),
        "large_body_collision": barrier_mean(body_distance, body_margin),
        "large_foot_collision": barrier_mean(foot_distance, foot_margin),
        "large_knee_shank_collision": barrier_mean(knee_shank_distance, link_margin),
        "large_terminal_risk": terminal_risk,
    }


__all__ = ["large_obstacle_losses", "small_object_losses"]
