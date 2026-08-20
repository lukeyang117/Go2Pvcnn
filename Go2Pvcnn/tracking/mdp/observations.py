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


def _plan_valid_mask(env, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    valid = getattr(manager, "step_plan_valid", getattr(manager, "plan_valid", None))
    if valid is None:
        valid = torch.ones(int(getattr(env, "num_envs")), dtype=torch.bool, device=device)
    return torch.as_tensor(valid, dtype=dtype, device=device).unsqueeze(-1)


def parallelism_plan_valid(env) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    valid = getattr(manager, "step_plan_valid", getattr(manager, "plan_valid", None))
    if valid is None:
        valid = torch.ones(int(getattr(env, "num_envs")), dtype=torch.float32, device=getattr(env, "device", "cpu"))
    return torch.as_tensor(valid, dtype=torch.float32, device=getattr(env, "device", "cpu")).unsqueeze(-1)


def parallelism_ref_joint_pos_rel_t(env) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    ref = manager.next_joint_pos
    default = _default_joint_pos(env).to(dtype=ref.dtype, device=ref.device)
    return (ref - default) * _plan_valid_mask(env, dtype=ref.dtype, device=ref.device)


def parallelism_ref_joint_vel_t(env) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    ref = manager.current_joint_vel
    return ref * _plan_valid_mask(env, dtype=ref.dtype, device=ref.device)


def parallelism_ref_root_pos_b_t(env) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    ref = manager.current_root_pos_b_policy
    return ref * _plan_valid_mask(env, dtype=ref.dtype, device=ref.device)


def parallelism_ref_root_rot_b_t(env) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    ref = manager.current_root_rot_b_policy
    return ref * _plan_valid_mask(env, dtype=ref.dtype, device=ref.device)
