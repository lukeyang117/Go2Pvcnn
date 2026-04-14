"""Reference reward helpers for trajectory-guided teacher experiments."""

from __future__ import annotations

import torch


def exponential_tracking_reward(error: torch.Tensor, sigma: float) -> torch.Tensor:
    """Convert a non-negative error tensor into an exponential reward."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return torch.exp(-error / (sigma**2))


def _broadcast_pair(current: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast a current/reference pair and keep comparison helpers compact."""
    current_b, reference_b = torch.broadcast_tensors(current, reference)
    return current_b, reference_b


def root_position_error(current_root_pos_w: torch.Tensor, reference_root_pos_w: torch.Tensor) -> torch.Tensor:
    """Return the per-sample L2 position error for the root pose."""
    current_root_pos_w, reference_root_pos_w = _broadcast_pair(current_root_pos_w, reference_root_pos_w)
    return torch.linalg.norm(current_root_pos_w - reference_root_pos_w, dim=-1)


def root_orientation_error(
    current_root_quat_w: torch.Tensor,
    reference_root_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return a quaternion-angle error for the root orientation."""
    current_root_quat_w, reference_root_quat_w = _broadcast_pair(current_root_quat_w, reference_root_quat_w)
    current_norm = torch.linalg.norm(current_root_quat_w, dim=-1, keepdim=True).clamp_min(1e-8)
    reference_norm = torch.linalg.norm(reference_root_quat_w, dim=-1, keepdim=True).clamp_min(1e-8)
    current_unit = current_root_quat_w / current_norm
    reference_unit = reference_root_quat_w / reference_norm
    dot = torch.sum(current_unit * reference_unit, dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.arccos(dot)


def compare_root_state(
    current_root_pos_w: torch.Tensor,
    reference_root_pos_w: torch.Tensor,
    current_root_quat_w: torch.Tensor,
    reference_root_quat_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare root position and orientation tensors."""
    return {
        "position_error": root_position_error(current_root_pos_w, reference_root_pos_w),
        "orientation_error": root_orientation_error(current_root_quat_w, reference_root_quat_w),
    }


def joint_position_error(current_joint_pos: torch.Tensor, reference_joint_pos: torch.Tensor) -> torch.Tensor:
    """Return the mean absolute joint error per sample."""
    current_joint_pos, reference_joint_pos = _broadcast_pair(current_joint_pos, reference_joint_pos)
    return torch.mean(torch.abs(current_joint_pos - reference_joint_pos), dim=-1)


def foot_position_error(current_foot_pos_root: torch.Tensor, reference_foot_pos_root: torch.Tensor) -> torch.Tensor:
    """Return the mean per-foot L2 error in the root frame."""
    current_foot_pos_root, reference_foot_pos_root = _broadcast_pair(current_foot_pos_root, reference_foot_pos_root)
    return torch.linalg.norm(current_foot_pos_root - reference_foot_pos_root, dim=-1).mean(dim=-1)


def contact_state_error(current_contact_state: torch.Tensor, reference_contact_state: torch.Tensor) -> torch.Tensor:
    """Return the mean contact mismatch rate per sample."""
    current_contact_state, reference_contact_state = _broadcast_pair(current_contact_state, reference_contact_state)
    return (current_contact_state != reference_contact_state).float().mean(dim=-1)


def touchdown_position_error(current_touchdown_pos_w: torch.Tensor, reference_touchdown_pos_w: torch.Tensor) -> torch.Tensor:
    """Return the mean touchdown position error per sample."""
    current_touchdown_pos_w, reference_touchdown_pos_w = _broadcast_pair(current_touchdown_pos_w, reference_touchdown_pos_w)
    return torch.linalg.norm(current_touchdown_pos_w - reference_touchdown_pos_w, dim=-1).mean(dim=-1)


def compare_reference_tensors(
    *,
    current_root_pos: torch.Tensor,
    reference_root_pos: torch.Tensor,
    current_root_quat: torch.Tensor,
    reference_root_quat: torch.Tensor,
    current_joint_pos: torch.Tensor,
    reference_joint_pos: torch.Tensor,
    current_foot_pos_root: torch.Tensor,
    reference_foot_pos_root: torch.Tensor,
    current_contact_state: torch.Tensor,
    reference_contact_state: torch.Tensor,
    current_touchdown_pos_w: torch.Tensor,
    reference_touchdown_pos_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare the full set of reference-tracking tensors."""
    root_state = compare_root_state(
        current_root_pos,
        reference_root_pos,
        current_root_quat,
        reference_root_quat,
    )
    return {
        "root_position_error": root_state["position_error"],
        "root_orientation_error": root_state["orientation_error"],
        "joint_pos_error": joint_position_error(current_joint_pos, reference_joint_pos),
        "foot_pos_root_error": foot_position_error(current_foot_pos_root, reference_foot_pos_root),
        "contact_state_error": contact_state_error(current_contact_state, reference_contact_state),
        "touchdown_pos_w_error": touchdown_position_error(current_touchdown_pos_w, reference_touchdown_pos_w),
    }


def _reference_horizon_steps(env) -> int:
    return int(getattr(env.cfg, "reference_trajectory_horizon", 50))


def _cache_matches_env(cache, env) -> bool:
    if cache is None or not cache.is_ready():
        return False
    rp = cache.root_pos_w
    if rp is None:
        return False
    h = cache.horizon_length()
    if h != _reference_horizon_steps(env):
        return False
    if rp.ndim == 3:
        return rp.shape[0] == env.num_envs
    return rp.ndim == 2 and env.num_envs == 1


def ensure_reference_cache(env):
    """Return the planner-owned reference cache for ``env``."""
    root = env.unwrapped
    manager = getattr(root, "_trajectory_manager", None)
    if manager is None:
        raise RuntimeError("planner-owned reference cache requires env.unwrapped._trajectory_manager")
    cache = manager.refresh_from_env(env)
    if cache is None or not cache.is_ready():
        raise RuntimeError("planner-owned reference cache is missing or not ready")
    root._trajectory_reference_cache = cache
    return cache


def _reference_indices(env, horizon: int) -> torch.Tensor:
    return (env.episode_length_buf.to(dtype=torch.long) % int(horizon)).to(env.device)


def _gather_reference_field(cache, name: str, frame_ids: torch.Tensor, env) -> torch.Tensor:
    """Index reference tensors for layout ``(H, ...)`` or ``(num_envs, H, ...)``."""
    t = getattr(cache, name)
    if t is None:
        raise RuntimeError(f"reference cache missing {name}")
    dev = env.device
    n = env.num_envs
    env_idx = torch.arange(n, device=dev, dtype=torch.long)
    if t.ndim == 2:
        return t[frame_ids].to(device=dev)
    if t.ndim == 3:
        return t[env_idx, frame_ids].to(device=dev)
    if t.ndim == 4:
        return t[env_idx, frame_ids].to(device=dev)
    raise RuntimeError(f"unexpected ndim {t.ndim} for {name}: {tuple(t.shape)}")


def _select_reference_frame(env):
    cache = ensure_reference_cache(env)
    horizon = cache.horizon_length()
    if horizon is None:
        raise RuntimeError("reference cache has no horizon")
    frame_ids = _reference_indices(env, horizon)
    return cache, frame_ids


def reference_root_pose_reward(
    env,
    pos_sigma: float = 0.5,
    rot_sigma: float = 0.5,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward current root pose staying close to the reference frame."""
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    cache, frame_ids = _select_reference_frame(env)
    asset: Articulation = env.scene[asset_cfg.name]
    ref_pos = _gather_reference_field(cache, "root_pos_w", frame_ids, env).to(dtype=asset.data.root_pos_w.dtype)
    ref_quat = _gather_reference_field(cache, "root_quat_w", frame_ids, env).to(dtype=asset.data.root_quat_w.dtype)
    pos_err = root_position_error(asset.data.root_pos_w, ref_pos)
    rot_err = root_orientation_error(asset.data.root_quat_w, ref_quat)
    return 0.5 * (
        exponential_tracking_reward(pos_err, sigma=pos_sigma)
        + exponential_tracking_reward(rot_err, sigma=rot_sigma)
    )


def reference_joint_pos_reward(
    env,
    sigma: float = 0.5,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward current joint positions staying close to the reference frame."""
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    cache, frame_ids = _select_reference_frame(env)
    asset: Articulation = env.scene[asset_cfg.name]
    ref_joint = _gather_reference_field(cache, "joint_angles", frame_ids, env).to(dtype=asset.data.joint_pos.dtype)
    err = joint_position_error(asset.data.joint_pos, ref_joint)
    return exponential_tracking_reward(err, sigma=sigma)


def _current_foot_positions_root(env, asset_cfg) -> torch.Tensor:
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    foot_rel_w = foot_pos_w - root_pos_w.unsqueeze(1)
    return math_utils.quat_rotate_inverse(
        root_quat_w.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1).reshape(-1, 4),
        foot_rel_w.reshape(-1, 3),
    ).reshape(env.num_envs, foot_pos_w.shape[1], 3)


def reference_foot_pos_reward(
    env,
    sigma: float = 0.5,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward current foot positions in root frame staying close to the reference frame."""
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=".*_foot")
    cache, frame_ids = _select_reference_frame(env)
    current_foot = _current_foot_positions_root(env, asset_cfg)
    ref_foot = _gather_reference_field(cache, "foot_pos_root", frame_ids, env).to(dtype=current_foot.dtype)
    err = foot_position_error(current_foot, ref_foot)
    return exponential_tracking_reward(err, sigma=sigma)


def reference_contact_reward(
    env,
    sigma: float = 0.5,
    sensor_cfg=None,
) -> torch.Tensor:
    """Reward contact state staying close to the reference frame."""
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor

    if sensor_cfg is None:
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
    cache, frame_ids = _select_reference_frame(env)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
    ref_contact = _gather_reference_field(cache, "contact_state", frame_ids, env)
    err = contact_state_error(current_contact, ref_contact)
    return exponential_tracking_reward(err, sigma=sigma)


def reference_touchdown_reward(
    env,
    sigma: float = 0.5,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward current foot positions staying close to planned touchdown positions."""
    from isaaclab.assets import Articulation
    from isaaclab.managers import SceneEntityCfg

    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", body_names=".*_foot")
    cache, frame_ids = _select_reference_frame(env)
    asset: Articulation = env.scene[asset_cfg.name]
    current_touchdown = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    ref_touchdown = _gather_reference_field(cache, "planned_touchdown_w", frame_ids, env).to(
        dtype=current_touchdown.dtype
    )
    err = touchdown_position_error(current_touchdown, ref_touchdown)
    return exponential_tracking_reward(err, sigma=sigma)


def zero_reference_reward(env, value: float = 0.0) -> torch.Tensor:
    """Fallback constant reward."""
    return torch.full((env.num_envs,), float(value), device=env.device)


__all__ = [
    "compare_contact_state",
    "compare_foot_pos_root",
    "compare_joint_pos",
    "compare_reference_tensors",
    "compare_root_state",
    "compare_touchdown_pos_w",
    "contact_state_error",
    "ensure_reference_cache",
    "exponential_tracking_reward",
    "foot_position_error",
    "joint_position_error",
    "reference_contact_reward",
    "reference_foot_pos_reward",
    "reference_joint_pos_reward",
    "reference_root_pose_reward",
    "reference_touchdown_reward",
    "root_orientation_error",
    "root_position_error",
    "touchdown_position_error",
    "zero_reference_reward",
]


def compare_joint_pos(current_joint_pos: torch.Tensor, reference_joint_pos: torch.Tensor) -> torch.Tensor:
    """Compatibility alias for joint comparison helpers."""
    return joint_position_error(current_joint_pos, reference_joint_pos)


def compare_foot_pos_root(
    current_foot_pos_root: torch.Tensor,
    reference_foot_pos_root: torch.Tensor,
) -> torch.Tensor:
    """Compatibility alias for root-frame foot comparison helpers."""
    return foot_position_error(current_foot_pos_root, reference_foot_pos_root)


def compare_contact_state(current_contact_state: torch.Tensor, reference_contact_state: torch.Tensor) -> torch.Tensor:
    """Compatibility alias for contact comparison helpers."""
    return contact_state_error(current_contact_state, reference_contact_state)


def compare_touchdown_pos_w(
    current_touchdown_pos_w: torch.Tensor,
    reference_touchdown_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Compatibility alias for touchdown comparison helpers."""
    return touchdown_position_error(current_touchdown_pos_w, reference_touchdown_pos_w)
