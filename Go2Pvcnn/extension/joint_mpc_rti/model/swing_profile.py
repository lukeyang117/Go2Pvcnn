"""Shared continuous swing-foot height profile."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg


def _quintic(value: Tensor) -> Tensor:
    return 10.0 * value.pow(3) - 15.0 * value.pow(4) + 6.0 * value.pow(5)


def swing_height_profile(
    lift_z: Tensor,
    apex_z: Tensor,
    touchdown_z: Tensor,
    tau: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Rise early, hold the apex, then descend with zero endpoint slopes."""
    ascent_end = float(cfg.nominal.swing_ascent_fraction)
    descent_start = float(cfg.nominal.swing_descent_fraction)
    ascent_tau = (tau / ascent_end).clamp(0.0, 1.0)
    descent_tau = ((tau - descent_start) / (1.0 - descent_start)).clamp(0.0, 1.0)
    ascent = lift_z + _quintic(ascent_tau) * (apex_z - lift_z)
    descent = apex_z + _quintic(descent_tau) * (touchdown_z - apex_z)
    landing_u = descent_tau
    landing_buffer = (
        4.0
        * landing_u
        * (1.0 - landing_u)
        * float(cfg.nominal.swing_landing_buffer_m)
    )
    descent = torch.maximum(descent, touchdown_z + landing_buffer)
    return torch.where(
        tau < ascent_end,
        ascent,
        torch.where(tau <= descent_start, apex_z, descent),
    )


def swing_xy_profile(
    lift_xy: Tensor,
    touchdown_xy: Tensor,
    command_axis: Tensor,
    tau: Tensor,
    *,
    crossing: Tensor,
    outward: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Use a delayed lateral shift for small-cross routes."""
    smooth = _quintic(tau.clamp(0.0, 1.0))
    bump = 64.0 * tau.pow(3) * (1.0 - tau).pow(3)
    standard = lift_xy + smooth * (touchdown_xy - lift_xy) + bump * outward

    delay = float(cfg.nominal.small_cross_lateral_start_fraction)
    crossing_tau = ((tau - delay) / (1.0 - delay)).clamp(0.0, 1.0)
    crossing_smooth = _quintic(crossing_tau)
    crossing_bump = 64.0 * crossing_tau.pow(3) * (1.0 - crossing_tau).pow(3)
    crossing_path = (
        lift_xy
        + crossing_smooth * (touchdown_xy - lift_xy)
        + crossing_bump * outward
    )
    return torch.where(crossing, crossing_path, standard)


def crossing_root_lift_offset(
    crossing: Tensor,
    tau: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    magnitude = (
        crossing[..., None, None].to(tau.dtype)
        * tau
        * float(cfg.nominal.small_cross_root_lift_m)
    )
    return torch.cat((torch.zeros_like(magnitude).expand(*magnitude.shape[:-1], 2), magnitude), dim=-1)
