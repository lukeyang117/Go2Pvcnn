"""Reference reward helpers for trajectory-guided teacher experiments."""

from __future__ import annotations

import os
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


def _trajectory_manager(env):
    return getattr(env.unwrapped, "_trajectory_manager", None)


def _uses_together_manager(manager) -> bool:
    return getattr(manager, "planner_backend", None) == "together"


def _reference_indices(env, horizon: int) -> torch.Tensor:
    return (env.episode_length_buf.to(dtype=torch.long) % int(horizon)).to(env.device)


def _gather_reference_field(cache, name: str, frame_ids: torch.Tensor, env) -> torch.Tensor:
    """Index reference tensors for layout ``(H, ...)`` or ``(num_envs, H, ...)``."""
    t = getattr(cache, name)
    if t is None:
        raise RuntimeError(f"reference cache missing {name}")
    dev = env.device
    n = env.num_envs
    idx_device = t.device
    frame_ids = frame_ids.to(device=idx_device, dtype=torch.long)
    env_idx = torch.arange(n, device=idx_device, dtype=torch.long)
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
    manager = _trajectory_manager(env)
    if _uses_together_manager(manager):
        frame_ids = manager.current_frame_ids()
    else:
        frame_ids = _reference_indices(env, horizon)
    return cache, frame_ids


def _reference_field(env, cache, name: str, frame_ids: torch.Tensor) -> torch.Tensor:
    manager = _trajectory_manager(env)
    if _uses_together_manager(manager):
        current = manager.current_reference()
        field = current.get(name)
        if field is None:
            raise RuntimeError(f"reference cache missing {name}")
        return field.to(device=env.device)
    return _gather_reference_field(cache, name, frame_ids, env)


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
    ref_pos = _reference_field(env, cache, "root_pos_w", frame_ids).to(dtype=asset.data.root_pos_w.dtype)
    ref_quat = _reference_field(env, cache, "root_quat_w", frame_ids).to(dtype=asset.data.root_quat_w.dtype)
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
    ref_joint = _reference_field(env, cache, "joint_angles", frame_ids).to(dtype=asset.data.joint_pos.dtype)
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
    ref_foot = _reference_field(env, cache, "foot_pos_root", frame_ids).to(dtype=current_foot.dtype)
    if os.environ.get("T302G_DEBUG_REF_REWARD_FINITE", "0").strip() == "1":
        if torch.any(~torch.isfinite(ref_foot)) or torch.any(~torch.isfinite(current_foot)):
            manager = _trajectory_manager(env)
            mask = None
            if manager is not None and hasattr(manager, "reference_reward_mask"):
                mask = manager.reference_reward_mask().detach().cpu().tolist()
            print(
                "[MPC][ReferenceFootRewardNonFinite] "
                f"current_has_nonfinite={bool(torch.any(~torch.isfinite(current_foot)).item())} "
                f"ref_has_nonfinite={bool(torch.any(~torch.isfinite(ref_foot)).item())} "
                f"frame_ids={frame_ids.detach().cpu().tolist()} "
                f"reward_mask={mask}",
                flush=True,
            )
    err = foot_position_error(current_foot, ref_foot)
    reward = exponential_tracking_reward(err, sigma=sigma)
    manager = _trajectory_manager(env)
    if manager is not None and hasattr(manager, "reference_reward_mask"):
        mask = manager.reference_reward_mask().to(device=reward.device, dtype=reward.dtype)
        reward = reward * mask
    return reward


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
    ref_contact = _reference_field(env, cache, "contact_state", frame_ids)
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
    ref_touchdown = _reference_field(env, cache, "planned_touchdown_w", frame_ids).to(dtype=current_touchdown.dtype)
    err = touchdown_position_error(current_touchdown, ref_touchdown)
    return exponential_tracking_reward(err, sigma=sigma)


def _scene_get(scene, name: str):
    try:
        return scene[name]
    except Exception:  # noqa: BLE001 - Isaac scene containers and tests are both duck-typed.
        return getattr(scene, name)


def _sensor_get(scene, name: str):
    sensors = getattr(scene, "sensors", None)
    if sensors is not None:
        try:
            return sensors[name]
        except Exception:  # noqa: BLE001
            if hasattr(sensors, name):
                return getattr(sensors, name)
    return _scene_get(scene, name)


def _normalize_body_name(name: str) -> str:
    return str(name).split("/")[-1].split(":")[-1].lower()


def _body_ids_and_names(robot, pattern: str):
    body_ids, body_names = robot.find_bodies(pattern)
    return [int(i) for i in body_ids], list(body_names)


def _leg_prefix(name: str) -> str:
    normalized = _normalize_body_name(name)
    return normalized.split("_", 1)[0]


def _scanner_terrain(scanner, *, device):
    from extension.batch_mpc_planner.terrain import MpcPlannerTerrain
    from extension.convention import extract_yaw_batch

    elevation = torch.as_tensor(scanner.data.elevation_map, dtype=torch.float32, device=device)
    semantic = torch.as_tensor(scanner.data.semantic_map, dtype=torch.long, device=device)
    pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
    size = getattr(pattern_cfg, "size", (1.5, 1.5))
    half_x = 0.5 * float(size[0])
    half_y = 0.5 * float(size[1])
    sensor_pos = torch.as_tensor(scanner.data.pos_w, dtype=torch.float32, device=device)
    sensor_quat = torch.as_tensor(scanner.data.quat_w, dtype=torch.float32, device=device)
    return MpcPlannerTerrain(
        height_map=elevation,
        semantic_map=semantic,
        world_x_range=(-half_x, half_x),
        world_y_range=(-half_y, half_y),
        sensor_pos_w=sensor_pos,
        sensor_yaw=extract_yaw_batch(sensor_quat),
    )


def swing_leg_collision_reward(
    env,
    clearance: float = 0.04,
    contact_force_threshold: float = 1.0,
    stance_weight: float = 0.25,
    swing_weight: float = 1.0,
    terrain_weight: float = 1.0,
    small_obstacle_weight: float = 2.0,
    large_obstacle_weight: float = 5.0,
    asset_cfg=None,
    sensor_cfg=None,
    scanner_cfg=None,
) -> torch.Tensor:
    """Penalize current leg bodies that intersect the semantic height field.

    This uses IsaacLab's current body/contact/scanner buffers. It intentionally
    does not use planner FK/IK or reference-cache state.
    """
    from extension.batch_mpc_planner.terrain import height_at, semantic_at

    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        asset_cfg = SceneEntityCfg("robot")
    if sensor_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
    if scanner_cfg is None:
        from isaaclab.managers import SceneEntityCfg

        scanner_cfg = SceneEntityCfg("semantic_height_scanner")

    device = torch.device(getattr(env, "device", "cpu"))
    robot = _scene_get(env.scene, asset_cfg.name)
    contact_sensor = _sensor_get(env.scene, sensor_cfg.name)
    scanner = _sensor_get(env.scene, scanner_cfg.name)

    leg_body_ids, leg_body_names = _body_ids_and_names(robot, ".*_(thigh|calf|foot)")
    if len(leg_body_ids) == 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    foot_body_ids, foot_body_names = _body_ids_and_names(robot, ".*_foot")
    leg_ids = torch.as_tensor(leg_body_ids, dtype=torch.long, device=device)
    body_pos = torch.as_tensor(robot.data.body_pos_w, dtype=torch.float32, device=device)[:, leg_ids, :]

    force_ids = getattr(sensor_cfg, "body_ids", None)
    if force_ids is None:
        force_ids = list(range(len(foot_body_ids)))
    force_ids_t = torch.as_tensor(force_ids, dtype=torch.long, device=device)
    forces = torch.as_tensor(contact_sensor.data.net_forces_w, dtype=torch.float32, device=device)[:, force_ids_t, :]
    foot_contact = torch.linalg.norm(forces, dim=-1) > float(contact_force_threshold)

    foot_prefix_to_col = {_leg_prefix(name): i for i, name in enumerate(foot_body_names)}
    leg_contact_cols = [foot_prefix_to_col.get(_leg_prefix(name), -1) for name in leg_body_names]
    valid_cols = torch.as_tensor([max(0, col) for col in leg_contact_cols], dtype=torch.long, device=device)
    leg_has_foot = torch.as_tensor([col >= 0 for col in leg_contact_cols], dtype=torch.bool, device=device)
    stance = foot_contact[:, valid_cols] & leg_has_foot.unsqueeze(0)
    leg_weight = torch.where(
        stance,
        torch.full_like(body_pos[..., 2], float(stance_weight)),
        torch.full_like(body_pos[..., 2], float(swing_weight)),
    )

    terrain = _scanner_terrain(scanner, device=device)
    body_xy = body_pos[..., :2]
    terrain_z = height_at(terrain, body_xy).to(dtype=body_pos.dtype, device=device)
    semantic = semantic_at(terrain, body_xy).to(device=device)
    clearance_violation = (terrain_z + float(clearance) - body_pos[..., 2]).clamp_min(0.0)
    semantic_weight = torch.ones_like(clearance_violation) * float(terrain_weight)
    semantic_weight = torch.where(semantic == 1, torch.full_like(semantic_weight, float(small_obstacle_weight)), semantic_weight)
    semantic_weight = torch.where(semantic >= 2, torch.full_like(semantic_weight, float(large_obstacle_weight)), semantic_weight)
    penalty = clearance_violation * semantic_weight * leg_weight
    return -penalty.sum(dim=1)


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
    "swing_leg_collision_reward",
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
