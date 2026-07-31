"""Observation terms for parallelism tracking."""

from __future__ import annotations

import torch

from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager


def _default_joint_pos(env) -> torch.Tensor:
    asset = env.scene["robot"]
    default = getattr(asset.data, "default_joint_pos", None)
    if default is None:
        return torch.zeros_like(asset.data.joint_pos)
    return torch.as_tensor(default, dtype=asset.data.joint_pos.dtype, device=asset.data.joint_pos.device)


def parallelism_ref_joint_pos_rel_t(env) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    ref = manager.current_joint_pos
    default = _default_joint_pos(env).to(dtype=ref.dtype, device=ref.device)
    return ref - default


def parallelism_ref_joint_vel_t(env) -> torch.Tensor:
    return get_parallelism_reference_manager(env).current_joint_vel


def parallelism_ref_root_lin_vel_b_t(env) -> torch.Tensor:
    return get_parallelism_reference_manager(env).current_root_lin_vel_b


def parallelism_ref_root_ang_vel_b_t(env) -> torch.Tensor:
    return get_parallelism_reference_manager(env).current_root_ang_vel_b
