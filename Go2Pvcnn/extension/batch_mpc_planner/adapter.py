"""Reference-cache adapter for batch MPC planner results."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.convention import euler_to_quat_batch
from extension.reference.cache import ReferenceTrajectoryCache


def _as_device_tensor(value, *, like: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    out = torch.as_tensor(value, device=like.device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out.contiguous()


def mpc_result_to_reference_cache(result) -> ReferenceTrajectoryCache:
    root_pos_w = torch.as_tensor(result.root_pos).contiguous()
    if root_pos_w.ndim != 3 or int(root_pos_w.shape[-1]) != 3:
        raise ValueError(f"root_pos must have shape (N,H,3), got {tuple(root_pos_w.shape)}")
    num_envs, horizon, _ = root_pos_w.shape
    root_quat_w = euler_to_quat_batch(result.root_rpy[..., 0], result.root_rpy[..., 1], result.root_rpy[..., 2])
    phase_row = torch.arange(horizon, dtype=torch.long, device=root_pos_w.device)
    phase_index = phase_row.unsqueeze(0).expand(num_envs, horizon).contiguous()
    valid_mask = torch.ones((num_envs, horizon), dtype=torch.bool, device=root_pos_w.device)
    foot_pos = _as_device_tensor(result.foot_pos, like=root_pos_w)
    foot_pos_root = foot_pos - root_pos_w.unsqueeze(2)
    return ReferenceTrajectoryCache(
        root_pos_w=root_pos_w,
        root_quat_w=_as_device_tensor(root_quat_w, like=root_pos_w),
        joint_angles=_as_device_tensor(result.joint_angles, like=root_pos_w),
        foot_pos_w=foot_pos,
        foot_pos_root=foot_pos_root,
        contact_state=_as_device_tensor(result.contact_state, like=root_pos_w, dtype=torch.bool),
        planned_touchdown_w=_as_device_tensor(result.planned_touchdown_w, like=root_pos_w),
        phase_index=phase_index,
        valid_mask=valid_mask,
    )


def result_new_ok_mask(result, *, num_envs: int, device: torch.device) -> Tensor:
    feasible = getattr(result, "feasible", None)
    safe_fallback = getattr(result, "safe_fallback", None)
    if feasible is None and safe_fallback is None:
        return torch.ones(num_envs, dtype=torch.bool, device=device)
    feasible_t = torch.ones(num_envs, dtype=torch.bool, device=device)
    if feasible is not None:
        feasible_t = torch.as_tensor(feasible, dtype=torch.bool, device=device)
    if safe_fallback is not None:
        fallback_t = torch.as_tensor(safe_fallback, dtype=torch.bool, device=device)
        feasible_t = torch.logical_and(feasible_t, torch.logical_not(fallback_t))
    return feasible_t


def standstill_cache_from_state(states, *, horizon: int) -> ReferenceTrajectoryCache:
    root_pos = torch.as_tensor(states.root_pos).contiguous()
    root_rpy = torch.as_tensor(states.root_rpy, device=root_pos.device)
    root_quat = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2])
    joint_angles = torch.as_tensor(states.joint_angles, device=root_pos.device).contiguous()
    foot_pos_w = torch.as_tensor(states.foot_pos, device=root_pos.device).contiguous()
    num_envs = int(root_pos.shape[0])
    phase_row = torch.arange(int(horizon), dtype=torch.long, device=root_pos.device)
    foot_pos_root = foot_pos_w - root_pos.unsqueeze(1)
    return ReferenceTrajectoryCache(
        root_pos_w=root_pos.unsqueeze(1).expand(num_envs, int(horizon), 3).contiguous(),
        root_quat_w=root_quat.unsqueeze(1).expand(num_envs, int(horizon), 4).contiguous(),
        joint_angles=joint_angles.unsqueeze(1).expand(num_envs, int(horizon), joint_angles.shape[-1]).contiguous(),
        foot_pos_w=foot_pos_w.unsqueeze(1).expand(num_envs, int(horizon), 4, 3).contiguous(),
        foot_pos_root=foot_pos_root.unsqueeze(1).expand(num_envs, int(horizon), 4, 3).contiguous(),
        contact_state=torch.ones((num_envs, int(horizon), 4), dtype=torch.bool, device=root_pos.device),
        planned_touchdown_w=foot_pos_w.unsqueeze(1).expand(num_envs, int(horizon), 4, 3).contiguous(),
        phase_index=phase_row.unsqueeze(0).expand(num_envs, int(horizon)).contiguous(),
        valid_mask=torch.ones((num_envs, int(horizon)), dtype=torch.bool, device=root_pos.device),
    )


def clone_reference_cache(cache: ReferenceTrajectoryCache) -> ReferenceTrajectoryCache:
    def cpy(x):
        return None if x is None else x.clone()

    return ReferenceTrajectoryCache(
        root_pos_w=cpy(cache.root_pos_w),
        root_quat_w=cpy(cache.root_quat_w),
        joint_angles=cpy(cache.joint_angles),
        foot_pos_w=cpy(cache.foot_pos_w),
        foot_pos_root=cpy(cache.foot_pos_root),
        contact_state=cpy(cache.contact_state),
        planned_touchdown_w=cpy(cache.planned_touchdown_w),
        phase_index=cpy(cache.phase_index),
        valid_mask=cpy(cache.valid_mask),
    )


def scatter_cache_rows(dst: ReferenceTrajectoryCache, src: ReferenceTrajectoryCache, env_ids: Tensor) -> None:
    idx = torch.as_tensor(env_ids, dtype=torch.long)
    if idx.numel() == 0:
        return
    fields = (
        "root_pos_w",
        "root_quat_w",
        "joint_angles",
        "foot_pos_w",
        "foot_pos_root",
        "contact_state",
        "planned_touchdown_w",
        "phase_index",
        "valid_mask",
    )
    for name in fields:
        dst_t = getattr(dst, name)
        src_t = getattr(src, name)
        if dst_t is None or src_t is None:
            raise ValueError(f"cache missing field {name}")
        dst_t.index_copy_(0, idx.to(device=dst_t.device), src_t.to(device=dst_t.device))


def blend_reference_caches(
    *,
    old_cache: ReferenceTrajectoryCache,
    new_cache: ReferenceTrajectoryCache,
    fallback_cache: ReferenceTrajectoryCache,
    replace_mask: Tensor,
    fallback_mask: Tensor,
) -> ReferenceTrajectoryCache:
    row = replace_mask.to(dtype=torch.bool, device=new_cache.root_pos_w.device)  # type: ignore[union-attr]
    fb = fallback_mask.to(dtype=torch.bool, device=row.device)
    row_3 = row.reshape(-1, 1, 1)
    fb_3 = fb.reshape(-1, 1, 1)
    row_4 = row.reshape(-1, 1, 1, 1)
    fb_4 = fb.reshape(-1, 1, 1, 1)
    row_2 = row.reshape(-1, 1)
    fb_2 = fb.reshape(-1, 1)
    return ReferenceTrajectoryCache(
        root_pos_w=torch.where(row_3, new_cache.root_pos_w, torch.where(fb_3, fallback_cache.root_pos_w, old_cache.root_pos_w)),
        root_quat_w=torch.where(row_3, new_cache.root_quat_w, torch.where(fb_3, fallback_cache.root_quat_w, old_cache.root_quat_w)),
        joint_angles=torch.where(row_3, new_cache.joint_angles, torch.where(fb_3, fallback_cache.joint_angles, old_cache.joint_angles)),
        foot_pos_w=torch.where(row_4, new_cache.foot_pos_w, torch.where(fb_4, fallback_cache.foot_pos_w, old_cache.foot_pos_w)),
        foot_pos_root=torch.where(row_4, new_cache.foot_pos_root, torch.where(fb_4, fallback_cache.foot_pos_root, old_cache.foot_pos_root)),
        contact_state=torch.where(row_3, new_cache.contact_state, torch.where(fb_3, fallback_cache.contact_state, old_cache.contact_state)),
        planned_touchdown_w=torch.where(
            row_4,
            new_cache.planned_touchdown_w,
            torch.where(fb_4, fallback_cache.planned_touchdown_w, old_cache.planned_touchdown_w),
        ),
        phase_index=torch.where(row_2, new_cache.phase_index, torch.where(fb_2, fallback_cache.phase_index, old_cache.phase_index)),
        valid_mask=torch.where(row_2, new_cache.valid_mask, torch.where(fb_2, fallback_cache.valid_mask, old_cache.valid_mask)),
    )


__all__ = [
    "blend_reference_caches",
    "clone_reference_cache",
    "mpc_result_to_reference_cache",
    "result_new_ok_mask",
    "scatter_cache_rows",
    "standstill_cache_from_state",
]
