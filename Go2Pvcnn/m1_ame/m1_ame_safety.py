"""M1-only finite-state and numerical safety helpers."""

from __future__ import annotations

from functools import wraps

import torch
from torch import Tensor


def finite_safe_tensor(values: Tensor, *, lower: float, upper: float) -> Tensor:
    """Replace non-finite values and clamp a tensor before it enters a reward/observation."""

    safe = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return safe.clamp(min=float(lower), max=float(upper))


def finite_safe_quaternion_wxyz(values: Tensor) -> Tensor:
    """Return finite unit quaternions, using identity for invalid/degenerate rows."""

    q = torch.as_tensor(values)
    safe = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
    norm = torch.linalg.vector_norm(safe, dim=-1, keepdim=True)
    valid = norm.squeeze(-1) > 1.0e-8
    normalized = safe / norm.clamp_min(1.0e-8)
    identity = torch.zeros_like(normalized)
    identity[..., 0] = 1.0
    return torch.where(valid.unsqueeze(-1), normalized, identity)


def _finite_rows(values: Tensor) -> Tensor:
    if values.ndim <= 1:
        return torch.isfinite(values)
    return torch.isfinite(values).reshape(values.shape[0], -1).all(dim=1)


def m1_invalid_state_mask(
    data,
    *,
    leg_joint_ids: tuple[int, ...] | list[int] | Tensor,
    wheel_joint_ids: tuple[int, ...] | list[int] | Tensor,
    leg_velocity_limit: float = 20.0,
    wheel_velocity_limit: float = 8.0,
) -> Tensor:
    """Return environments whose articulation state is non-finite or physically divergent."""

    joint_pos = torch.as_tensor(data.joint_pos)
    device = joint_pos.device
    invalid = torch.zeros(joint_pos.shape[0], dtype=torch.bool, device=device)
    for name in (
        "joint_pos",
        "joint_vel",
        "joint_acc",
        "applied_torque",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "body_lin_vel_w",
    ):
        value = getattr(data, name, None)
        if value is not None:
            invalid |= ~_finite_rows(torch.as_tensor(value, device=device))

    joint_vel = torch.as_tensor(data.joint_vel, device=device)
    leg_ids = torch.as_tensor(leg_joint_ids, dtype=torch.long, device=device)
    wheel_ids = torch.as_tensor(wheel_joint_ids, dtype=torch.long, device=device)
    if leg_ids.numel():
        invalid |= torch.any(joint_vel.index_select(-1, leg_ids).abs() > float(leg_velocity_limit), dim=-1)
    if wheel_ids.numel():
        invalid |= torch.any(joint_vel.index_select(-1, wheel_ids).abs() > float(wheel_velocity_limit), dim=-1)
    return invalid


def m1_finite_reward(func):
    """Wrap an M1 reward so invalid-state steps cannot poison episode sums."""

    @wraps(func)
    def wrapped(env, *args, **kwargs):
        value = torch.as_tensor(func(env, *args, **kwargs))
        value = finite_safe_tensor(value, lower=-1.0e6, upper=1.0e6)
        invalid = getattr(env, "_m1_invalid_state_last", None)
        if invalid is not None and value.ndim > 0 and value.shape[0] == invalid.shape[0]:
            invalid = invalid.to(device=value.device, dtype=torch.bool)
            value = torch.where(invalid, torch.zeros_like(value), value)
        return value

    return wrapped


__all__ = ["finite_safe_quaternion_wxyz", "finite_safe_tensor", "m1_finite_reward", "m1_invalid_state_mask"]


def m1_joint_pos_rel_safe(env, asset_cfg):
    from isaaclab.envs import mdp as isaac_mdp

    return finite_safe_tensor(isaac_mdp.joint_pos_rel(env, asset_cfg=asset_cfg), lower=-25.0, upper=25.0)


def m1_joint_vel_rel_safe(env, asset_cfg):
    from isaaclab.envs import mdp as isaac_mdp

    return finite_safe_tensor(isaac_mdp.joint_vel_rel(env, asset_cfg=asset_cfg), lower=-25.0, upper=25.0)


__all__ += ["m1_joint_pos_rel_safe", "m1_joint_vel_rel_safe"]
