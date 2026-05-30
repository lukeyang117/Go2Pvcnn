"""Device-preserving cache adapter for the together trajectory backend."""

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


def _expand_touchdowns(touchdown: Tensor, *, num_envs: int, horizon: int) -> Tensor:
    if touchdown.ndim == 2:
        return touchdown.reshape(1, 1, 4, 3).expand(num_envs, horizon, 4, 3).contiguous()
    if touchdown.ndim == 3:
        return touchdown.unsqueeze(1).expand(num_envs, horizon, 4, 3).contiguous()
    if touchdown.ndim == 4:
        return touchdown.contiguous()
    raise ValueError(f"planned_touchdown_w must have ndim 2, 3, or 4; got {touchdown.ndim}")


def together_result_to_reference_cache(result) -> ReferenceTrajectoryCache:
    """Convert a full-batch together planner result without canonical CPU migration."""

    root_pos_value = getattr(result, "root_pos_w", getattr(result, "root_pos", None))
    if root_pos_value is None:
        raise ValueError("planner result missing root_pos/root_pos_w")
    root_pos_w = torch.as_tensor(root_pos_value).contiguous()
    if root_pos_w.ndim != 3 or int(root_pos_w.shape[-1]) != 3:
        raise ValueError(f"root_pos_w must have shape (N,H,3); got {tuple(root_pos_w.shape)}")
    num_envs = int(root_pos_w.shape[0])
    horizon = int(root_pos_w.shape[1])
    phase_row = torch.arange(horizon, dtype=torch.long, device=root_pos_w.device)
    phase_index = phase_row.unsqueeze(0).expand(num_envs, horizon).contiguous()
    valid_mask = torch.ones((num_envs, horizon), dtype=torch.bool, device=root_pos_w.device)
    root_quat_value = getattr(result, "root_quat_w", None)
    if root_quat_value is None:
        root_rpy = _as_device_tensor(result.root_rpy, like=root_pos_w)
        root_quat_value = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2])
    foot_pos_root_value = getattr(result, "foot_pos_root", None)
    foot_pos_w_value = getattr(result, "foot_pos_w", getattr(result, "foot_pos", None))
    if foot_pos_root_value is None:
        if foot_pos_w_value is None:
            raise ValueError("planner result missing foot_pos/foot_pos_w/foot_pos_root")
        foot_pos_w_value = _as_device_tensor(foot_pos_w_value, like=root_pos_w)
        foot_pos_root_value = foot_pos_w_value - root_pos_w.unsqueeze(2)
    elif foot_pos_w_value is None:
        foot_pos_w_value = root_pos_w.unsqueeze(2) + _as_device_tensor(foot_pos_root_value, like=root_pos_w)
    else:
        foot_pos_w_value = _as_device_tensor(foot_pos_w_value, like=root_pos_w)
    touchdown_value = getattr(result, "planned_touchdown_w", None)
    if touchdown_value is None:
        touchdown_value = result.touchdown_seq[:, :, 0, :]
    touchdown = _as_device_tensor(touchdown_value, like=root_pos_w)

    return ReferenceTrajectoryCache(
        root_pos_w=root_pos_w,
        root_quat_w=_as_device_tensor(root_quat_value, like=root_pos_w),
        joint_angles=_as_device_tensor(result.joint_angles, like=root_pos_w),
        foot_pos_w=_as_device_tensor(foot_pos_w_value, like=root_pos_w),
        foot_pos_root=_as_device_tensor(foot_pos_root_value, like=root_pos_w),
        contact_state=_as_device_tensor(result.contact_state, like=root_pos_w, dtype=torch.bool),
        planned_touchdown_w=_expand_touchdowns(touchdown, num_envs=num_envs, horizon=horizon),
        phase_index=phase_index,
        valid_mask=valid_mask,
    )


def result_new_ok_mask(result, *, num_envs: int, device: torch.device) -> Tensor:
    """Return ``feasible OR safe_fallback`` with legacy-result compatibility."""

    feasible = getattr(result, "feasible", None)
    safe_fallback = getattr(result, "safe_fallback", None)
    if feasible is None and safe_fallback is None:
        return torch.ones(num_envs, dtype=torch.bool, device=device)
    feasible_t = torch.zeros(num_envs, dtype=torch.bool, device=device)
    fallback_t = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if feasible is not None:
        feasible_t = torch.as_tensor(feasible, dtype=torch.bool, device=device)
    if safe_fallback is not None:
        fallback_t = torch.as_tensor(safe_fallback, dtype=torch.bool, device=device)
    return torch.logical_or(feasible_t, fallback_t)


def standstill_cache_from_state(states, *, horizon: int) -> ReferenceTrajectoryCache:
    """Build a current-state standstill cache on the state tensor device."""

    root_pos = torch.as_tensor(states.root_pos).contiguous()
    root_quat_value = getattr(states, "root_quat", None)
    if root_quat_value is None:
        root_rpy = torch.as_tensor(states.root_rpy, device=root_pos.device)
        root_quat_value = euler_to_quat_batch(root_rpy[..., 0], root_rpy[..., 1], root_rpy[..., 2])
    root_quat = torch.as_tensor(root_quat_value, device=root_pos.device).contiguous()
    joint_angles = torch.as_tensor(states.joint_angles, device=root_pos.device).contiguous()
    foot_pos_w = torch.as_tensor(states.foot_pos, device=root_pos.device).contiguous()
    num_envs = int(root_pos.shape[0])
    phase_row = torch.arange(int(horizon), dtype=torch.long, device=root_pos.device)
    foot_pos_root = foot_pos_w - root_pos.unsqueeze(1)

    return ReferenceTrajectoryCache(
        root_pos_w=root_pos.unsqueeze(1).expand(num_envs, int(horizon), 3).contiguous(),
        root_quat_w=root_quat.unsqueeze(1).expand(num_envs, int(horizon), 4).contiguous(),
        joint_angles=joint_angles.unsqueeze(1).expand(num_envs, int(horizon), 12).contiguous(),
        foot_pos_w=foot_pos_w.unsqueeze(1).expand(num_envs, int(horizon), 4, 3).contiguous(),
        foot_pos_root=foot_pos_root.unsqueeze(1).expand(num_envs, int(horizon), 4, 3).contiguous(),
        contact_state=torch.ones((num_envs, int(horizon), 4), dtype=torch.bool, device=root_pos.device),
        planned_touchdown_w=foot_pos_w.unsqueeze(1).expand(num_envs, int(horizon), 4, 3).contiguous(),
        phase_index=phase_row.unsqueeze(0).expand(num_envs, int(horizon)).contiguous(),
        valid_mask=torch.ones((num_envs, int(horizon)), dtype=torch.bool, device=root_pos.device),
    )


def blend_reference_caches(
    *,
    old_cache: ReferenceTrajectoryCache,
    new_cache: ReferenceTrajectoryCache,
    fallback_cache: ReferenceTrajectoryCache,
    replace_mask: Tensor,
    fallback_mask: Tensor,
) -> ReferenceTrajectoryCache:
    """Blend cache rows using fixed-shape ``torch.where`` masks."""

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
    "result_new_ok_mask",
    "standstill_cache_from_state",
    "together_result_to_reference_cache",
]
