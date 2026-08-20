"""Reward terms for parallelism tracking."""

from __future__ import annotations

import torch

try:
    from isaaclab.managers import SceneEntityCfg
except Exception:  # noqa: BLE001 - allows lightweight unit tests without an Isaac app.
    class SceneEntityCfg:  # type: ignore[no-redef]
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            for key, value in kwargs.items():
                setattr(self, key, value)

from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager
from extension.parallelism.collision import official_collision_mask
from extension.parallelism.kinematics import fk_go2
from extension.parallelism.terrain import query_height_semantic_valid


_LEG_NAMES = ("FL", "FR", "RL", "RR")
_FOOT_NAMES = tuple(f"{leg}_foot" for leg in _LEG_NAMES)


def _parallelism_plan_valid(env, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    manager = get_parallelism_reference_manager(env)
    valid = getattr(manager, "step_plan_valid", getattr(manager, "plan_valid", None))
    if valid is None:
        valid = torch.ones(int(getattr(env, "num_envs")), dtype=torch.bool, device=device)
    return torch.as_tensor(valid, dtype=dtype, device=device)


def _active_small_obstacle_safe_mask(
    semantic: torch.Tensor,
    height: torch.Tensor,
    foot_z: torch.Tensor,
    valid: torch.Tensor,
    collision: torch.Tensor,
    contact: torch.Tensor,
    *,
    touchdown_tolerance_m: float,
) -> torch.Tensor:
    """Return active feet touching a small semantic obstacle without collision."""

    return (
        (~contact)
        & valid
        & semantic.eq(1)
        & (torch.abs(foot_z - height) <= float(touchdown_tolerance_m))
        & (~collision)
    )


def _active_collision_penalty(collision: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
    """Return one penalty event when any active leg has a geometry collision."""

    return (collision & (~contact)).any(dim=-1).to(dtype=torch.float32)


def _ensure_obstacle_stats(env, *, device: torch.device, dtype: torch.dtype) -> None:
    count = int(env.scene["robot"].data.root_pos_w.shape[0])
    if not hasattr(env, "_parallelism_obstacle_collision_sum"):
        env._parallelism_obstacle_collision_sum = torch.zeros(count, dtype=dtype, device=device)
        env._parallelism_obstacle_collision_count = torch.zeros(count, dtype=torch.long, device=device)
        env._parallelism_obstacle_small_foot_sum = torch.zeros(count, dtype=dtype, device=device)
        env._parallelism_obstacle_small_foot_count = torch.zeros(count, dtype=torch.long, device=device)
        env._parallelism_obstacle_safe_foot_sum = torch.zeros(count, dtype=dtype, device=device)
        env._parallelism_obstacle_safe_foot_count = torch.zeros(count, dtype=torch.long, device=device)
        env._parallelism_obstacle_standstill_sum = torch.zeros(count, dtype=dtype, device=device)
        env._parallelism_obstacle_valid_sum = torch.zeros(count, dtype=dtype, device=device)
        env._parallelism_obstacle_step_count = torch.zeros(count, dtype=torch.long, device=device)


def _policy_parallelism_collision_by_leg(env) -> torch.Tensor:
    """Run the planner's official geometry collision test on the live policy pose."""

    manager = get_parallelism_reference_manager(env)
    state = manager._state(torch.arange(manager.num_envs, device=manager.device, dtype=torch.long))
    geometry = fk_go2(
        state.root_pos_w,
        state.root_rpy_w,
        state.joint_pos,
        capsule_samples=int(manager.cfg.capsule_samples),
    )
    batch = int(state.root_pos_w.shape[0])
    leg_count = int(geometry.foot_pos_w.shape[-2])
    candidate_count = 1

    def _expand_pose(value: torch.Tensor) -> torch.Tensor:
        return value[:, None, None].expand(batch, leg_count, candidate_count, *value.shape[1:])

    def _expand_samples(value: torch.Tensor) -> torch.Tensor:
        return value[:, None, None].expand(batch, leg_count, candidate_count, *value.shape[1:])

    geometry = type(geometry)(
        hip_pos_w=_expand_pose(geometry.hip_pos_w),
        foot_pos_w=_expand_pose(geometry.foot_pos_w),
        knee_pos_w=_expand_pose(geometry.knee_pos_w),
        calf_samples_w=_expand_samples(geometry.calf_samples_w),
        thigh_samples_w=_expand_samples(geometry.thigh_samples_w),
        thigh_pos_w=_expand_pose(geometry.thigh_pos_w),
        thigh_rot_w=_expand_pose(geometry.thigh_rot_w),
        calf_pos_w=_expand_pose(geometry.calf_pos_w),
        calf_rot_w=_expand_pose(geometry.calf_rot_w),
        foot_rot_w=_expand_pose(geometry.foot_rot_w),
    )
    _, collision_bits = official_collision_mask(manager._terrain(state.root_pos_w), geometry, manager.cfg)
    return collision_bits.any(dim=-1).squeeze(-1)


def _update_obstacle_stats(
    env,
    *,
    collision_event: torch.Tensor,
    small_foot_event: torch.Tensor | None = None,
    safe_foot_event: torch.Tensor | None = None,
) -> None:
    manager = get_parallelism_reference_manager(env)
    _ensure_obstacle_stats(
        env,
        device=collision_event.device,
        dtype=collision_event.dtype,
    )
    valid = _parallelism_plan_valid(env, dtype=collision_event.dtype, device=collision_event.device)
    invalid = 1.0 - valid
    env._parallelism_obstacle_collision_sum += collision_event
    env._parallelism_obstacle_collision_count += 1
    env._parallelism_obstacle_standstill_sum += manager.standstill_latched.to(dtype=collision_event.dtype)
    env._parallelism_obstacle_valid_sum += valid
    env._parallelism_obstacle_step_count += 1
    _ensure_obstacle_split_stats(env, device=collision_event.device, dtype=collision_event.dtype)
    env._parallelism_obstacle_valid_collision_sum += collision_event * valid
    env._parallelism_obstacle_invalid_collision_sum += collision_event * invalid
    standstill = manager.standstill_latched.to(dtype=collision_event.dtype)
    env._parallelism_obstacle_valid_standstill_sum += standstill * valid
    env._parallelism_obstacle_invalid_standstill_sum += standstill * invalid
    env._parallelism_obstacle_valid_collision_count += valid.to(dtype=torch.long)
    env._parallelism_obstacle_invalid_collision_count += invalid.to(dtype=torch.long)
    env._parallelism_obstacle_valid_step_count += valid.to(dtype=torch.long)
    env._parallelism_obstacle_invalid_step_count += invalid.to(dtype=torch.long)
    if small_foot_event is not None:
        env._parallelism_obstacle_small_foot_sum += small_foot_event
        env._parallelism_obstacle_small_foot_count += 1
        env._parallelism_obstacle_valid_small_foot_sum += small_foot_event * valid
        env._parallelism_obstacle_invalid_small_foot_sum += small_foot_event * invalid
        env._parallelism_obstacle_valid_small_foot_count += valid.to(dtype=torch.long)
        env._parallelism_obstacle_invalid_small_foot_count += invalid.to(dtype=torch.long)
    if safe_foot_event is not None:
        env._parallelism_obstacle_safe_foot_sum += safe_foot_event
        env._parallelism_obstacle_safe_foot_count += 1
        env._parallelism_obstacle_valid_safe_foot_sum += safe_foot_event * valid
        env._parallelism_obstacle_invalid_safe_foot_sum += safe_foot_event * invalid


def _ensure_obstacle_split_stats(env, *, device: torch.device, dtype: torch.dtype) -> None:
    count = int(env.scene["robot"].data.root_pos_w.shape[0])
    if hasattr(env, "_parallelism_obstacle_valid_collision_sum"):
        return
    scalar = lambda: torch.zeros(count, dtype=dtype, device=device)
    integer = lambda: torch.zeros(count, dtype=torch.long, device=device)
    env._parallelism_obstacle_valid_collision_sum = scalar()
    env._parallelism_obstacle_invalid_collision_sum = scalar()
    env._parallelism_obstacle_valid_small_foot_sum = scalar()
    env._parallelism_obstacle_invalid_small_foot_sum = scalar()
    env._parallelism_obstacle_valid_safe_foot_sum = scalar()
    env._parallelism_obstacle_invalid_safe_foot_sum = scalar()
    env._parallelism_obstacle_valid_standstill_sum = scalar()
    env._parallelism_obstacle_invalid_standstill_sum = scalar()
    env._parallelism_obstacle_valid_collision_count = integer()
    env._parallelism_obstacle_invalid_collision_count = integer()
    env._parallelism_obstacle_valid_small_foot_count = integer()
    env._parallelism_obstacle_invalid_small_foot_count = integer()
    env._parallelism_obstacle_valid_step_count = integer()
    env._parallelism_obstacle_invalid_step_count = integer()


def parallelism_geometry_collision_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scanner_cfg: SceneEntityCfg = SceneEntityCfg("semantic_height_scanner"),
) -> torch.Tensor:
    """Return the raw active-leg collision event.

    The reward configuration supplies the negative penalty weight. Keeping the
    function value positive avoids applying the penalty sign twice in Isaac
    Lab's ``value = func(...) * weight * dt`` path.
    """

    _ = asset_cfg, scanner_cfg
    manager = get_parallelism_reference_manager(env)
    collision_by_leg = _policy_parallelism_collision_by_leg(env)
    contact = torch.as_tensor(manager.current_contact_state, dtype=torch.bool, device=collision_by_leg.device)
    event = _active_collision_penalty(collision_by_leg, contact)
    _update_obstacle_stats(env, collision_event=event)
    return event


def active_swing_foot_on_small_obstacle_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scanner_cfg: SceneEntityCfg = SceneEntityCfg("semantic_height_scanner"),
    touchdown_tolerance_m: float = 0.04,
) -> torch.Tensor:
    """Reward an active swing foot that safely lands on a small obstacle."""

    _ = asset_cfg, scanner_cfg
    manager = get_parallelism_reference_manager(env)
    state = manager._state(torch.arange(manager.num_envs, device=manager.device, dtype=torch.long))
    terrain = manager._terrain(state.root_pos_w)
    query = query_height_semantic_valid(terrain, state.foot_pos_w[..., :2])
    collision_by_leg = _policy_parallelism_collision_by_leg(env)
    contact = torch.as_tensor(manager.current_contact_state, dtype=torch.bool, device=query.semantic.device)
    safe = _active_small_obstacle_safe_mask(
        query.semantic,
        query.height,
        state.foot_pos_w[..., 2],
        query.valid,
        collision_by_leg,
        contact,
        touchdown_tolerance_m=touchdown_tolerance_m,
    )
    small_foot = ((~contact) & query.valid & query.semantic.eq(1)).any(dim=-1).to(dtype=query.height.dtype)
    safe_event = safe.any(dim=-1).to(dtype=query.height.dtype)
    _update_obstacle_stats(
        env,
        collision_event=torch.zeros_like(safe_event),
        small_foot_event=small_foot,
        safe_foot_event=safe_event,
    )
    return safe_event


def parallelism_obstacle_episode_metrics(env) -> dict[str, torch.Tensor]:
    """Return episode-normalized obstacle diagnostics for the tracking env hook."""

    if not hasattr(env, "_parallelism_obstacle_step_count"):
        return {}
    steps = env._parallelism_obstacle_step_count.clamp_min(1).to(dtype=env._parallelism_obstacle_collision_sum.dtype)
    collision_count = env._parallelism_obstacle_collision_count.clamp_min(1).to(dtype=steps.dtype)
    small_count = env._parallelism_obstacle_small_foot_count.clamp_min(1).to(dtype=steps.dtype)
    _ensure_obstacle_split_stats(
        env,
        device=env._parallelism_obstacle_collision_sum.device,
        dtype=env._parallelism_obstacle_collision_sum.dtype,
    )
    valid_collision_count = env._parallelism_obstacle_valid_collision_count.clamp_min(1).to(dtype=steps.dtype)
    invalid_collision_count = env._parallelism_obstacle_invalid_collision_count.clamp_min(1).to(dtype=steps.dtype)
    valid_small_count = env._parallelism_obstacle_valid_small_foot_count.clamp_min(1).to(dtype=steps.dtype)
    invalid_small_count = env._parallelism_obstacle_invalid_small_foot_count.clamp_min(1).to(dtype=steps.dtype)
    valid_steps = env._parallelism_obstacle_valid_step_count.clamp_min(1).to(dtype=steps.dtype)
    invalid_steps = env._parallelism_obstacle_invalid_step_count.clamp_min(1).to(dtype=steps.dtype)
    return {
        "geometry_collision_ratio": env._parallelism_obstacle_collision_sum / collision_count,
        "active_swing_foot_on_small_ratio": env._parallelism_obstacle_small_foot_sum / small_count,
        "active_swing_foot_on_small_no_collision_ratio": env._parallelism_obstacle_safe_foot_sum / small_count,
        "standstill_ratio": env._parallelism_obstacle_standstill_sum / steps,
        "reference_valid_ratio": env._parallelism_obstacle_valid_sum / steps,
        "valid_geometry_collision_ratio": env._parallelism_obstacle_valid_collision_sum / valid_collision_count,
        "invalid_geometry_collision_ratio": env._parallelism_obstacle_invalid_collision_sum / invalid_collision_count,
        "valid_standstill_ratio": env._parallelism_obstacle_valid_standstill_sum / valid_steps,
        "invalid_standstill_ratio": env._parallelism_obstacle_invalid_standstill_sum / invalid_steps,
        "valid_active_swing_foot_on_small_ratio": env._parallelism_obstacle_valid_small_foot_sum / valid_small_count,
        "invalid_active_swing_foot_on_small_ratio": env._parallelism_obstacle_invalid_small_foot_sum / invalid_small_count,
        "valid_active_swing_foot_on_small_no_collision_ratio": env._parallelism_obstacle_valid_safe_foot_sum / valid_small_count,
        "invalid_active_swing_foot_on_small_no_collision_ratio": env._parallelism_obstacle_invalid_safe_foot_sum / invalid_small_count,
        "valid_step_count": env._parallelism_obstacle_valid_step_count.to(dtype=steps.dtype),
        "invalid_step_count": env._parallelism_obstacle_invalid_step_count.to(dtype=steps.dtype),
        "valid_step_ratio": env._parallelism_obstacle_valid_step_count.to(dtype=steps.dtype) / steps,
        "invalid_step_ratio": env._parallelism_obstacle_invalid_step_count.to(dtype=steps.dtype) / steps,
    }


def reset_parallelism_obstacle_stats(env, env_ids: torch.Tensor) -> None:
    """Reset obstacle episode accumulators for selected environments."""

    if not hasattr(env, "_parallelism_obstacle_step_count"):
        return
    for name in (
        "_parallelism_obstacle_collision_sum",
        "_parallelism_obstacle_small_foot_sum",
        "_parallelism_obstacle_safe_foot_sum",
        "_parallelism_obstacle_standstill_sum",
        "_parallelism_obstacle_valid_sum",
        "_parallelism_obstacle_valid_collision_sum",
        "_parallelism_obstacle_invalid_collision_sum",
        "_parallelism_obstacle_valid_small_foot_sum",
        "_parallelism_obstacle_invalid_small_foot_sum",
        "_parallelism_obstacle_valid_safe_foot_sum",
        "_parallelism_obstacle_invalid_safe_foot_sum",
        "_parallelism_obstacle_valid_standstill_sum",
        "_parallelism_obstacle_invalid_standstill_sum",
    ):
        getattr(env, name)[env_ids] = 0
    for name in (
        "_parallelism_obstacle_collision_count",
        "_parallelism_obstacle_small_foot_count",
        "_parallelism_obstacle_safe_foot_count",
        "_parallelism_obstacle_step_count",
        "_parallelism_obstacle_valid_collision_count",
        "_parallelism_obstacle_invalid_collision_count",
        "_parallelism_obstacle_valid_small_foot_count",
        "_parallelism_obstacle_invalid_small_foot_count",
        "_parallelism_obstacle_valid_step_count",
        "_parallelism_obstacle_invalid_step_count",
    ):
        getattr(env, name)[env_ids] = 0


def _gaussian_error_reward(error: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-torch.sum(torch.square(error), dim=-1) / (float(std) * float(std)))


def _tolerance(error: torch.Tensor, tracking_tolerance: float) -> torch.Tensor:
    if float(tracking_tolerance) <= 0.0:
        return error
    return torch.sign(error) * torch.clamp(torch.abs(error) - float(tracking_tolerance), min=0.0)


def _quat_to_matrix_wxyz(quat: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(quat)
    w, x, y, z = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    matrix = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    matrix[..., 0, 0] = 1 - two * (y * y + z * z)
    matrix[..., 0, 1] = two * (x * y - w * z)
    matrix[..., 0, 2] = two * (x * z + w * y)
    matrix[..., 1, 0] = two * (x * y + w * z)
    matrix[..., 1, 1] = 1 - two * (x * x + z * z)
    matrix[..., 1, 2] = two * (y * z - w * x)
    matrix[..., 2, 0] = two * (x * z - w * y)
    matrix[..., 2, 1] = two * (y * z + w * x)
    matrix[..., 2, 2] = 1 - two * (x * x + y * y)
    return matrix


def _points_to_root_frame(points_w: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    matrix_w = _quat_to_matrix_wxyz(root_quat_w)
    rel_w = points_w - root_pos_w[:, None, :]
    return torch.matmul(matrix_w.transpose(-1, -2)[:, None], rel_w.unsqueeze(-1)).squeeze(-1)


def _actual_foot_pos_w(asset, asset_cfg: SceneEntityCfg, ref: torch.Tensor) -> torch.Tensor:
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is not None and not (isinstance(body_ids, slice) and body_ids == slice(None)):
        body_pos = asset.data.body_pos_w[:, body_ids, :]
    else:
        body_pos = asset.data.body_pos_w[:, -4:, :]
    return torch.as_tensor(body_pos, dtype=ref.dtype, device=ref.device)


def _normalize_articulation_name(name: str) -> str:
    normalized = str(name).split("/")[-1]
    return normalized.split(":")[-1].lower()


def _tracking_foot_body_ids(env, asset, *, device: torch.device) -> torch.Tensor:
    cached = getattr(env, "_parallelism_tracking_foot_body_ids", None)
    if cached is not None:
        return torch.as_tensor(cached, dtype=torch.long, device=device)

    body_pos = torch.as_tensor(asset.data.body_pos_w)
    if hasattr(asset, "find_bodies"):
        body_ids, body_names = asset.find_bodies(".*_foot")
        source = {_normalize_articulation_name(name): int(body_id) for body_id, name in zip(body_ids, body_names)}
        try:
            ordered = [source[_normalize_articulation_name(name)] for name in _FOOT_NAMES]
        except KeyError as exc:
            raise ValueError(f"Missing Go2 foot body for tracking: {exc.args[0]}") from exc
    elif int(body_pos.shape[1]) == 4:
        ordered = [0, 1, 2, 3]
    else:
        raise ValueError("Tracking metrics require named FL/FR/RL/RR foot bodies")

    result = torch.tensor(ordered, dtype=torch.long, device=device)
    env._parallelism_tracking_foot_body_ids = result
    return result


def _tracking_foot_pos_w(env, asset, ref: torch.Tensor) -> torch.Tensor:
    body_pos = torch.as_tensor(asset.data.body_pos_w, dtype=ref.dtype, device=ref.device)
    body_ids = _tracking_foot_body_ids(env, asset, device=ref.device)
    return body_pos.index_select(1, body_ids)


def _tracking_joint_leg_ids(env, asset, *, device: torch.device) -> torch.Tensor:
    cached = getattr(env, "_parallelism_tracking_joint_leg_ids", None)
    if cached is not None:
        return torch.as_tensor(cached, dtype=torch.long, device=device)

    joint_names = tuple(getattr(asset, "joint_names", ()) or ())
    if joint_names:
        source = {_normalize_articulation_name(name): index for index, name in enumerate(joint_names)}
        ordered = [
            [
                source[_normalize_articulation_name(f"{leg}_hip_joint")],
                source[_normalize_articulation_name(f"{leg}_thigh_joint")],
                source[_normalize_articulation_name(f"{leg}_calf_joint")],
            ]
            for leg in _LEG_NAMES
        ]
    elif int(asset.data.joint_pos.shape[1]) == 12:
        ordered = [[3 * leg + joint for joint in range(3)] for leg in range(4)]
    else:
        raise ValueError("Tracking metrics require named Go2 joints or exactly 12 planner-ordered joints")

    result = torch.tensor(ordered, dtype=torch.long, device=device)
    env._parallelism_tracking_joint_leg_ids = result
    return result


def _current_parallelism_tracking_errors(env, asset_cfg: SceneEntityCfg) -> dict[str, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    manager = get_parallelism_reference_manager(env)
    ref_pos_b = manager.current_root_pos_b_policy
    ref_rot_b = manager.current_root_rot_b_policy
    ref_joint = manager.step_joint_pos
    actual_joint = torch.as_tensor(asset.data.joint_pos, dtype=ref_joint.dtype, device=ref_joint.device)
    joint_abs_error = torch.abs(actual_joint - ref_joint)
    joint_leg_ids = _tracking_joint_leg_ids(env, asset, device=ref_joint.device)
    joint_max_error_per_leg = joint_abs_error.index_select(1, joint_leg_ids.flatten()).reshape(-1, 4, 3).amax(dim=-1)
    if hasattr(asset.data, "body_pos_w"):
        ref_foot = manager.step_foot_pos_w
        actual_foot = _tracking_foot_pos_w(env, asset, ref_foot)
        root_pos = torch.as_tensor(asset.data.root_pos_w, dtype=ref_foot.dtype, device=ref_foot.device)
        root_quat = torch.as_tensor(asset.data.root_quat_w, dtype=ref_foot.dtype, device=ref_foot.device)
        foot_delta = _points_to_root_frame(actual_foot, root_pos, root_quat) - _points_to_root_frame(
            ref_foot, root_pos, root_quat
        )
        foot_abs_error = torch.linalg.vector_norm(foot_delta, dim=-1)
        foot_z_abs_error = torch.abs(foot_delta[..., 2])
    else:
        foot_abs_error = torch.zeros(
            actual_joint.shape[0],
            4,
            dtype=actual_joint.dtype,
            device=actual_joint.device,
        )
        foot_z_abs_error = torch.zeros_like(foot_abs_error)
    contact = getattr(manager, "current_contact_state", torch.ones_like(foot_abs_error, dtype=torch.bool))
    swing_mask = ~torch.as_tensor(contact, dtype=torch.bool, device=foot_abs_error.device)
    return {
        "root_pos_error": torch.linalg.vector_norm(ref_pos_b, dim=-1),
        "root_rot_error": torch.linalg.vector_norm(ref_rot_b, dim=-1),
        "joint_mean_error": torch.mean(joint_abs_error, dim=-1),
        "joint_max_error": torch.max(joint_abs_error, dim=-1).values,
        "joint_max_error_per_leg": joint_max_error_per_leg,
        "foot_mean_error": torch.mean(foot_abs_error, dim=-1),
        "foot_max_error": torch.max(foot_abs_error, dim=-1).values,
        "foot_error_per_leg": foot_abs_error,
        "foot_z_error_per_leg": foot_z_abs_error,
        "swing_mask": swing_mask,
    }


_TRACKING_SPLIT_SUFFIXES = (
    "joint_mean_sum",
    "joint_max",
    "foot_mean_sum",
    "foot_max",
    "root_pos_sum",
    "root_rot_sum",
    "error_frames",
    "active_swing_foot_sum",
    "active_swing_foot_count",
    "active_swing_foot_max",
    "active_swing_foot_z_sum",
    "active_swing_foot_z_max",
    "swing_foot_sum_per_leg",
    "swing_foot_count_per_leg",
    "swing_foot_max_per_leg",
    "swing_foot_z_sum_per_leg",
    "joint_max_per_leg",
)


def _ensure_tracking_split_stats(env, errors: dict[str, torch.Tensor]) -> None:
    count = int(errors["joint_mean_error"].shape[0])
    dtype = errors["joint_mean_error"].dtype
    device = errors["joint_mean_error"].device
    per_leg_shape = (count, 4)
    for prefix in ("valid", "invalid"):
        for suffix in _TRACKING_SPLIT_SUFFIXES:
            name = f"_parallelism_tracking_{prefix}_{suffix}"
            if hasattr(env, name):
                continue
            if suffix.endswith("_per_leg"):
                if "count" in suffix:
                    value = torch.zeros(per_leg_shape, dtype=torch.long, device=device)
                else:
                    value = torch.zeros(per_leg_shape, dtype=dtype, device=device)
            elif suffix in ("error_frames", "active_swing_foot_count"):
                value = torch.zeros(count, dtype=torch.long, device=device)
            else:
                value = torch.zeros(count, dtype=dtype, device=device)
            setattr(env, name, value)


def _current_command_tracking_errors(env, *, command_name: str = "base_velocity") -> tuple[torch.Tensor, torch.Tensor]:
    count = int(getattr(env, "num_envs", 0))
    device = torch.device(getattr(env, "device", "cpu"))
    asset = env.scene.get("robot") if hasattr(env.scene, "get") else env.scene["robot"]
    data = getattr(asset, "data", None)
    command_manager = getattr(env, "command_manager", None)
    if data is None or command_manager is None or not hasattr(command_manager, "get_command"):
        zeros = torch.zeros(count, dtype=torch.float32, device=device)
        return zeros, zeros
    command = torch.as_tensor(command_manager.get_command(command_name), device=device)
    root_lin_vel_b = getattr(data, "root_lin_vel_b", None)
    root_ang_vel_b = getattr(data, "root_ang_vel_b", None)
    if root_lin_vel_b is None or root_ang_vel_b is None:
        zeros = torch.zeros(count, dtype=command.dtype, device=command.device)
        return zeros, zeros
    lin_vel = torch.as_tensor(root_lin_vel_b, dtype=command.dtype, device=command.device)
    ang_vel = torch.as_tensor(root_ang_vel_b, dtype=command.dtype, device=command.device)
    return torch.linalg.vector_norm(command[:, :2] - lin_vel[:, :2], dim=-1), torch.abs(command[:, 2] - ang_vel[:, 2])


def _ensure_command_split_stats(env, *, count: int, dtype: torch.dtype, device: torch.device) -> None:
    for prefix in ("valid", "invalid"):
        for suffix in ("command_lin_vel_sum", "command_ang_vel_sum", "command_frames"):
            name = f"_parallelism_tracking_{prefix}_{suffix}"
            if hasattr(env, name):
                continue
            value_dtype = torch.long if suffix == "command_frames" else dtype
            setattr(env, name, torch.zeros(count, dtype=value_dtype, device=device))


def _accumulate_command_tracking_stats(
    env,
    lin_error: torch.Tensor,
    ang_error: torch.Tensor,
    update_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    _ensure_command_split_stats(
        env,
        count=int(lin_error.shape[0]),
        dtype=lin_error.dtype,
        device=lin_error.device,
    )
    for prefix, bucket_mask in (("valid", update_mask & valid_mask), ("invalid", update_mask & ~valid_mask)):
        getattr(env, f"_parallelism_tracking_{prefix}_command_lin_vel_sum").add_(
            torch.where(bucket_mask, lin_error, torch.zeros_like(lin_error))
        )
        getattr(env, f"_parallelism_tracking_{prefix}_command_ang_vel_sum").add_(
            torch.where(bucket_mask, ang_error, torch.zeros_like(ang_error))
        )
        getattr(env, f"_parallelism_tracking_{prefix}_command_frames").add_(bucket_mask.to(dtype=torch.long))


def _accumulate_tracking_split_stats(
    env,
    errors: dict[str, torch.Tensor],
    update_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    _ensure_tracking_split_stats(env, errors)
    for prefix, bucket_mask in (("valid", update_mask & valid_mask), ("invalid", update_mask & ~valid_mask)):
        def scalar(name: str) -> torch.Tensor:
            return getattr(env, f"_parallelism_tracking_{prefix}_{name}")

        scalar("joint_mean_sum").add_(torch.where(bucket_mask, errors["joint_mean_error"], torch.zeros_like(errors["joint_mean_error"])))
        scalar("foot_mean_sum").add_(torch.where(bucket_mask, errors["foot_mean_error"], torch.zeros_like(errors["foot_mean_error"])))
        scalar("root_pos_sum").add_(torch.where(bucket_mask, errors["root_pos_error"], torch.zeros_like(errors["root_pos_error"])))
        scalar("root_rot_sum").add_(torch.where(bucket_mask, errors["root_rot_error"], torch.zeros_like(errors["root_rot_error"])))
        scalar("joint_max").copy_(torch.maximum(scalar("joint_max"), torch.where(bucket_mask, errors["joint_max_error"], torch.zeros_like(errors["joint_max_error"]))))
        scalar("foot_max").copy_(torch.maximum(scalar("foot_max"), torch.where(bucket_mask, errors["foot_max_error"], torch.zeros_like(errors["foot_max_error"]))))
        scalar("error_frames").add_(bucket_mask.to(dtype=torch.long))

        active_mask = bucket_mask[:, None] & errors["swing_mask"]
        active_foot_error = torch.where(active_mask, errors["foot_error_per_leg"], torch.zeros_like(errors["foot_error_per_leg"]))
        active_foot_z_error = torch.where(active_mask, errors["foot_z_error_per_leg"], torch.zeros_like(errors["foot_z_error_per_leg"]))
        scalar("active_swing_foot_sum").add_(active_foot_error.sum(dim=-1))
        scalar("active_swing_foot_count").add_(active_mask.sum(dim=-1).to(dtype=torch.long))
        scalar("active_swing_foot_max").copy_(torch.maximum(scalar("active_swing_foot_max"), active_foot_error.amax(dim=-1)))
        scalar("active_swing_foot_z_sum").add_(active_foot_z_error.sum(dim=-1))
        scalar("active_swing_foot_z_max").copy_(torch.maximum(scalar("active_swing_foot_z_max"), active_foot_z_error.amax(dim=-1)))
        scalar("swing_foot_sum_per_leg").add_(active_foot_error)
        scalar("swing_foot_count_per_leg").add_(active_mask.to(dtype=torch.long))
        scalar("swing_foot_max_per_leg").copy_(torch.maximum(scalar("swing_foot_max_per_leg"), active_foot_error))
        scalar("swing_foot_z_sum_per_leg").add_(active_foot_z_error)
        scalar("joint_max_per_leg").copy_(torch.maximum(
            scalar("joint_max_per_leg"),
            torch.where(bucket_mask[:, None], errors["joint_max_error_per_leg"], torch.zeros_like(errors["joint_max_error_per_leg"])),
        ))


def update_parallelism_tracking_error_stats(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Accumulate current-phase tracking errors once per environment step."""

    step_token = getattr(env, "common_step_counter", None)
    if step_token is not None:
        step_token = int(step_token)
    cached_token = getattr(env, "_parallelism_tracking_error_cache_token", None)
    cached_errors = getattr(env, "_parallelism_tracking_error_cache", None)
    if step_token is not None and cached_token == step_token and cached_errors is not None:
        errors = cached_errors
    else:
        errors = _current_parallelism_tracking_errors(env, asset_cfg)
        if step_token is not None:
            env._parallelism_tracking_error_cache_token = step_token
            env._parallelism_tracking_error_cache = errors

    device = errors["root_pos_error"].device
    count = int(getattr(env, "num_envs", errors["root_pos_error"].shape[0]))
    episode_step = getattr(env, "episode_length_buf", None)
    if episode_step is None:
        episode_step = getattr(env, "_parallelism_tracking_fallback_step", 0)
        step = torch.full((count,), int(episode_step), dtype=torch.long, device=device)
        env._parallelism_tracking_fallback_step = int(episode_step) + 1
    else:
        step = torch.as_tensor(episode_step, dtype=torch.long, device=device).reshape(count)

    last_step = getattr(
        env,
        "_parallelism_tracking_last_step",
        torch.full((count,), -1, dtype=torch.long, device=device),
    )
    last_step = torch.as_tensor(last_step, dtype=torch.long, device=device)
    update_mask = step != last_step
    valid_mask = _parallelism_plan_valid(env, dtype=torch.bool, device=device)
    command_lin_error, command_ang_error = _current_command_tracking_errors(env)

    if not hasattr(env, "_parallelism_tracking_joint_mean_sum"):
        env._parallelism_tracking_joint_mean_sum = torch.zeros(count, dtype=errors["joint_mean_error"].dtype, device=device)
        env._parallelism_tracking_joint_max = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_foot_mean_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_foot_max = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_root_pos_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_root_rot_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_error_frames = torch.zeros(count, dtype=torch.long, device=device)
        env._parallelism_tracking_active_swing_foot_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_active_swing_foot_count = torch.zeros(count, dtype=torch.long, device=device)
        env._parallelism_tracking_active_swing_foot_max = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_active_swing_foot_z_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_active_swing_foot_z_max = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_swing_foot_sum_per_leg = torch.zeros(count, 4, dtype=errors["joint_mean_error"].dtype, device=device)
        env._parallelism_tracking_swing_foot_count_per_leg = torch.zeros(count, 4, dtype=torch.long, device=device)
        env._parallelism_tracking_swing_foot_max_per_leg = torch.zeros_like(
            env._parallelism_tracking_swing_foot_sum_per_leg
        )
        env._parallelism_tracking_swing_foot_z_sum_per_leg = torch.zeros_like(
            env._parallelism_tracking_swing_foot_sum_per_leg
        )
        env._parallelism_tracking_joint_max_per_leg = torch.zeros_like(
            env._parallelism_tracking_swing_foot_sum_per_leg
        )

    _accumulate_tracking_split_stats(env, errors, update_mask, valid_mask)
    _accumulate_command_tracking_stats(env, command_lin_error, command_ang_error, update_mask, valid_mask)
    legacy_update_mask = update_mask & valid_mask

    env._parallelism_tracking_joint_mean_sum += torch.where(
        legacy_update_mask, errors["joint_mean_error"], torch.zeros_like(errors["joint_mean_error"])
    )
    env._parallelism_tracking_joint_max = torch.maximum(
        env._parallelism_tracking_joint_max,
        torch.where(legacy_update_mask, errors["joint_max_error"], torch.zeros_like(errors["joint_max_error"])),
    )
    env._parallelism_tracking_foot_mean_sum += torch.where(
        legacy_update_mask, errors["foot_mean_error"], torch.zeros_like(errors["foot_mean_error"])
    )
    env._parallelism_tracking_foot_max = torch.maximum(
        env._parallelism_tracking_foot_max,
        torch.where(legacy_update_mask, errors["foot_max_error"], torch.zeros_like(errors["foot_max_error"])),
    )
    env._parallelism_tracking_root_pos_sum += torch.where(
        legacy_update_mask, errors["root_pos_error"], torch.zeros_like(errors["root_pos_error"])
    )
    env._parallelism_tracking_root_rot_sum += torch.where(
        legacy_update_mask, errors["root_rot_error"], torch.zeros_like(errors["root_rot_error"])
    )
    active_mask = legacy_update_mask[:, None] & errors["swing_mask"]
    active_foot_error = torch.where(active_mask, errors["foot_error_per_leg"], torch.zeros_like(errors["foot_error_per_leg"]))
    active_foot_z_error = torch.where(
        active_mask,
        errors["foot_z_error_per_leg"],
        torch.zeros_like(errors["foot_z_error_per_leg"]),
    )
    env._parallelism_tracking_active_swing_foot_sum += active_foot_error.sum(dim=-1)
    env._parallelism_tracking_active_swing_foot_count += active_mask.sum(dim=-1)
    env._parallelism_tracking_active_swing_foot_max = torch.maximum(
        env._parallelism_tracking_active_swing_foot_max,
        active_foot_error.amax(dim=-1),
    )
    env._parallelism_tracking_active_swing_foot_z_sum += active_foot_z_error.sum(dim=-1)
    env._parallelism_tracking_active_swing_foot_z_max = torch.maximum(
        env._parallelism_tracking_active_swing_foot_z_max,
        active_foot_z_error.amax(dim=-1),
    )
    env._parallelism_tracking_swing_foot_sum_per_leg += active_foot_error
    env._parallelism_tracking_swing_foot_count_per_leg += active_mask.to(dtype=torch.long)
    env._parallelism_tracking_swing_foot_max_per_leg = torch.maximum(
        env._parallelism_tracking_swing_foot_max_per_leg,
        active_foot_error,
    )
    env._parallelism_tracking_swing_foot_z_sum_per_leg += active_foot_z_error
    env._parallelism_tracking_joint_max_per_leg = torch.maximum(
        env._parallelism_tracking_joint_max_per_leg,
        torch.where(
            legacy_update_mask[:, None],
            errors["joint_max_error_per_leg"],
            torch.zeros_like(errors["joint_max_error_per_leg"]),
        ),
    )
    env._parallelism_tracking_error_frames += legacy_update_mask.to(dtype=torch.long)
    env._parallelism_tracking_last_step = torch.where(update_mask, step, last_step)
    return {**errors, **_parallelism_tracking_episode_stats(env)}


def _parallelism_tracking_bucket_stats(env, prefix: str) -> dict[str, torch.Tensor]:
    dtype = getattr(env, f"_parallelism_tracking_{prefix}_joint_mean_sum").dtype
    scalar_zero = torch.zeros_like(getattr(env, f"_parallelism_tracking_{prefix}_joint_mean_sum"))
    per_leg_zero = torch.zeros(scalar_zero.shape[0], 4, dtype=dtype, device=scalar_zero.device)
    frames = getattr(env, f"_parallelism_tracking_{prefix}_error_frames").clamp_min(1).to(dtype=dtype)
    command_frames = getattr(env, f"_parallelism_tracking_{prefix}_command_frames").clamp_min(1).to(dtype=dtype)
    swing_count = getattr(env, f"_parallelism_tracking_{prefix}_active_swing_foot_count").clamp_min(1).to(dtype=dtype)
    swing_count_per_leg = getattr(env, f"_parallelism_tracking_{prefix}_swing_foot_count_per_leg").clamp_min(1).to(dtype=dtype)
    return {
        f"{prefix}_episode_joint_mean_error": getattr(env, f"_parallelism_tracking_{prefix}_joint_mean_sum") / frames,
        f"{prefix}_episode_joint_max_error": getattr(env, f"_parallelism_tracking_{prefix}_joint_max"),
        f"{prefix}_episode_foot_mean_error": getattr(env, f"_parallelism_tracking_{prefix}_foot_mean_sum") / frames,
        f"{prefix}_episode_foot_max_error": getattr(env, f"_parallelism_tracking_{prefix}_foot_max"),
        f"{prefix}_episode_root_pos_error": getattr(env, f"_parallelism_tracking_{prefix}_root_pos_sum") / frames,
        f"{prefix}_episode_root_rot_error": getattr(env, f"_parallelism_tracking_{prefix}_root_rot_sum") / frames,
        f"{prefix}_episode_reference_frame_count": getattr(env, f"_parallelism_tracking_{prefix}_error_frames").to(dtype=dtype),
        f"{prefix}_episode_active_swing_foot_mean_error": getattr(env, f"_parallelism_tracking_{prefix}_active_swing_foot_sum") / swing_count,
        f"{prefix}_episode_active_swing_foot_max_error": getattr(env, f"_parallelism_tracking_{prefix}_active_swing_foot_max"),
        f"{prefix}_episode_active_swing_foot_z_mean_error": getattr(env, f"_parallelism_tracking_{prefix}_active_swing_foot_z_sum") / swing_count,
        f"{prefix}_episode_active_swing_foot_z_max_error": getattr(env, f"_parallelism_tracking_{prefix}_active_swing_foot_z_max"),
        f"{prefix}_episode_swing_foot_mean_error_per_leg": getattr(env, f"_parallelism_tracking_{prefix}_swing_foot_sum_per_leg") / swing_count_per_leg,
        f"{prefix}_episode_swing_foot_max_error_per_leg": getattr(env, f"_parallelism_tracking_{prefix}_swing_foot_max_per_leg"),
        f"{prefix}_episode_swing_foot_z_mean_error_per_leg": getattr(env, f"_parallelism_tracking_{prefix}_swing_foot_z_sum_per_leg") / swing_count_per_leg,
        f"{prefix}_episode_joint_max_error_per_leg": getattr(env, f"_parallelism_tracking_{prefix}_joint_max_per_leg"),
        f"{prefix}_episode_command_lin_vel_error": getattr(env, f"_parallelism_tracking_{prefix}_command_lin_vel_sum") / command_frames,
        f"{prefix}_episode_command_ang_vel_error": getattr(env, f"_parallelism_tracking_{prefix}_command_ang_vel_sum") / command_frames,
        f"{prefix}_episode_command_frame_count": getattr(env, f"_parallelism_tracking_{prefix}_command_frames").to(dtype=dtype),
    }


def _parallelism_tracking_episode_stats(env) -> dict[str, torch.Tensor]:
    dtype = env._parallelism_tracking_joint_mean_sum.dtype
    if not hasattr(env, "_parallelism_tracking_valid_joint_mean_sum"):
        _ensure_tracking_split_stats(
            env,
            {"joint_mean_error": env._parallelism_tracking_joint_mean_sum},
        )
    _ensure_command_split_stats(
        env,
        count=int(env._parallelism_tracking_joint_mean_sum.shape[0]),
        dtype=dtype,
        device=env._parallelism_tracking_joint_mean_sum.device,
    )
    frames = env._parallelism_tracking_error_frames.clamp_min(1).to(dtype=dtype)
    scalar_zero = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
    per_leg_zero = torch.zeros(scalar_zero.shape[0], 4, dtype=dtype, device=scalar_zero.device)
    swing_foot_sum = getattr(env, "_parallelism_tracking_active_swing_foot_sum", scalar_zero)
    swing_foot_count = getattr(
        env,
        "_parallelism_tracking_active_swing_foot_count",
        torch.zeros_like(scalar_zero, dtype=torch.long),
    )
    swing_foot_max = getattr(env, "_parallelism_tracking_active_swing_foot_max", scalar_zero)
    swing_foot_z_sum = getattr(env, "_parallelism_tracking_active_swing_foot_z_sum", scalar_zero)
    swing_foot_z_max = getattr(env, "_parallelism_tracking_active_swing_foot_z_max", scalar_zero)
    swing_foot_sum_per_leg = getattr(env, "_parallelism_tracking_swing_foot_sum_per_leg", per_leg_zero)
    swing_foot_count_per_leg = getattr(
        env,
        "_parallelism_tracking_swing_foot_count_per_leg",
        torch.zeros_like(per_leg_zero, dtype=torch.long),
    )
    swing_foot_max_per_leg = getattr(env, "_parallelism_tracking_swing_foot_max_per_leg", per_leg_zero)
    swing_foot_z_sum_per_leg = getattr(env, "_parallelism_tracking_swing_foot_z_sum_per_leg", per_leg_zero)
    joint_max_per_leg = getattr(env, "_parallelism_tracking_joint_max_per_leg", per_leg_zero)
    foot_mean_sum = getattr(env, "_parallelism_tracking_foot_mean_sum", scalar_zero)
    foot_max = getattr(env, "_parallelism_tracking_foot_max", scalar_zero)
    swing_count = swing_foot_count.clamp_min(1).to(dtype=dtype)
    swing_count_per_leg = swing_foot_count_per_leg.clamp_min(1).to(dtype=dtype)
    return {
        "episode_joint_mean_error": env._parallelism_tracking_joint_mean_sum / frames,
        "episode_joint_max_error": env._parallelism_tracking_joint_max,
        "episode_foot_mean_error": foot_mean_sum / frames,
        "episode_foot_max_error": foot_max,
        "episode_root_pos_error": env._parallelism_tracking_root_pos_sum / frames,
        "episode_root_rot_error": env._parallelism_tracking_root_rot_sum / frames,
        "episode_active_swing_foot_mean_error": swing_foot_sum / swing_count,
        "episode_active_swing_foot_max_error": swing_foot_max,
        "episode_active_swing_foot_z_mean_error": swing_foot_z_sum / swing_count,
        "episode_active_swing_foot_z_max_error": swing_foot_z_max,
        "episode_swing_foot_mean_error_per_leg": swing_foot_sum_per_leg / swing_count_per_leg,
        "episode_swing_foot_max_error_per_leg": swing_foot_max_per_leg,
        "episode_swing_foot_z_mean_error_per_leg": swing_foot_z_sum_per_leg / swing_count_per_leg,
        "episode_joint_max_error_per_leg": joint_max_per_leg,
        **_parallelism_tracking_bucket_stats(env, "valid"),
        **_parallelism_tracking_bucket_stats(env, "invalid"),
    }


def reset_parallelism_tracking_error_stats(env, env_ids: torch.Tensor) -> None:
    """Reset accumulated tracking statistics for selected environments."""

    if not hasattr(env, "_parallelism_tracking_joint_mean_sum"):
        return
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env._parallelism_tracking_joint_mean_sum.device)
    for name in (
        "_parallelism_tracking_joint_mean_sum",
        "_parallelism_tracking_joint_max",
        "_parallelism_tracking_foot_mean_sum",
        "_parallelism_tracking_foot_max",
        "_parallelism_tracking_root_pos_sum",
        "_parallelism_tracking_root_rot_sum",
        "_parallelism_tracking_error_frames",
        "_parallelism_tracking_active_swing_foot_sum",
        "_parallelism_tracking_active_swing_foot_count",
        "_parallelism_tracking_active_swing_foot_max",
        "_parallelism_tracking_active_swing_foot_z_sum",
        "_parallelism_tracking_active_swing_foot_z_max",
        "_parallelism_tracking_swing_foot_sum_per_leg",
        "_parallelism_tracking_swing_foot_count_per_leg",
        "_parallelism_tracking_swing_foot_max_per_leg",
        "_parallelism_tracking_swing_foot_z_sum_per_leg",
        "_parallelism_tracking_joint_max_per_leg",
    ):
        if hasattr(env, name):
            getattr(env, name)[ids] = 0
    for prefix in ("valid", "invalid"):
        for suffix in _TRACKING_SPLIT_SUFFIXES:
            name = f"_parallelism_tracking_{prefix}_{suffix}"
            if hasattr(env, name):
                getattr(env, name)[ids] = 0
        for suffix in ("command_lin_vel_sum", "command_ang_vel_sum", "command_frames"):
            name = f"_parallelism_tracking_{prefix}_{suffix}"
            if hasattr(env, name):
                getattr(env, name)[ids] = 0
    if hasattr(env, "_parallelism_tracking_last_step"):
        env._parallelism_tracking_last_step[ids] = -1
    if hasattr(env, "_parallelism_tracking_error_cache"):
        env._parallelism_tracking_error_cache_token = None


def parallelism_tracking_episode_errors(env) -> dict[str, torch.Tensor]:
    """Read stored episode tracking statistics without adding a reset-frame sample."""

    if not hasattr(env, "_parallelism_tracking_joint_mean_sum"):
        return update_parallelism_tracking_error_stats(env)
    return _parallelism_tracking_episode_stats(env)


def reference_joint_pos_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.35,
    tracking_tolerance: float = 0.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).step_joint_pos
    actual = torch.as_tensor(asset.data.joint_pos, dtype=ref.dtype, device=ref.device)
    update_parallelism_tracking_error_stats(env, asset_cfg)
    error = _tolerance(actual - ref, tracking_tolerance)
    return _gaussian_error_reward(error, std) * _parallelism_plan_valid(
        env, dtype=error.dtype, device=error.device
    )


def reference_joint_vel_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 2.0,
    tracking_tolerance: float = 0.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).step_joint_vel
    actual = torch.as_tensor(asset.data.joint_vel, dtype=ref.dtype, device=ref.device)
    update_parallelism_tracking_error_stats(env, asset_cfg)
    error = _tolerance(actual - ref, tracking_tolerance)
    return _gaussian_error_reward(error, std) * _parallelism_plan_valid(
        env, dtype=error.dtype, device=error.device
    )


def reference_joint_max_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.8,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).step_joint_pos
    actual = torch.as_tensor(asset.data.joint_pos, dtype=ref.dtype, device=ref.device)
    worst_error = torch.abs(actual - ref).amax(dim=-1)
    reward = torch.exp(-torch.square(worst_error / float(std)))
    return reward * _parallelism_plan_valid(env, dtype=reward.dtype, device=reward.device)


def reference_root_pos_reward(env, std: float = 0.12) -> torch.Tensor:
    """Primary reward for tracking the next Parallelism root position."""

    manager = get_parallelism_reference_manager(env)
    error = manager.current_root_pos_b_policy
    update_parallelism_tracking_error_stats(env)
    return _gaussian_error_reward(error, std) * _parallelism_plan_valid(
        env, dtype=error.dtype, device=error.device
    )


def reference_root_rot_reward(env, std: float = 0.30) -> torch.Tensor:
    """Primary reward for tracking the next Parallelism root orientation."""

    manager = get_parallelism_reference_manager(env)
    error = manager.current_root_rot_b_policy
    update_parallelism_tracking_error_stats(env)
    return _gaussian_error_reward(error, std) * _parallelism_plan_valid(
        env, dtype=error.dtype, device=error.device
    )


def reference_foot_pos_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    std: float = 0.12,
    stance_weight: float = 1.0,
    swing_weight: float = 2.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    manager = get_parallelism_reference_manager(env)
    ref_w = manager.step_foot_pos_w
    actual_w = _tracking_foot_pos_w(env, asset, ref_w)
    root_pos = torch.as_tensor(asset.data.root_pos_w, dtype=ref_w.dtype, device=ref_w.device)
    root_quat = torch.as_tensor(asset.data.root_quat_w, dtype=ref_w.dtype, device=ref_w.device)
    ref_b = _points_to_root_frame(ref_w, root_pos, root_quat)
    actual_b = _points_to_root_frame(actual_w, root_pos, root_quat)
    contact = getattr(manager, "current_contact_state", torch.ones(ref_w.shape[:2], dtype=torch.bool, device=ref_w.device))
    contact = torch.as_tensor(contact, dtype=torch.bool, device=ref_w.device)
    weight = torch.where(
        contact,
        torch.full(contact.shape, float(stance_weight), dtype=ref_w.dtype, device=ref_w.device),
        torch.full(contact.shape, float(swing_weight), dtype=ref_w.dtype, device=ref_w.device),
    )
    squared_error = torch.sum(torch.square(actual_b - ref_b), dim=-1)
    normalized_error = torch.sum(weight * squared_error, dim=-1) / torch.clamp_min(torch.sum(weight, dim=-1), 1.0e-6)
    reward = torch.exp(-normalized_error / (float(std) * float(std)))
    return reward * _parallelism_plan_valid(env, dtype=reward.dtype, device=reward.device)


def reference_active_swing_foot_max_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    std: float = 0.12,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    manager = get_parallelism_reference_manager(env)
    ref_w = manager.step_foot_pos_w
    actual_w = _tracking_foot_pos_w(env, asset, ref_w)
    root_pos = torch.as_tensor(asset.data.root_pos_w, dtype=ref_w.dtype, device=ref_w.device)
    root_quat = torch.as_tensor(asset.data.root_quat_w, dtype=ref_w.dtype, device=ref_w.device)
    foot_error = torch.linalg.vector_norm(
        _points_to_root_frame(actual_w, root_pos, root_quat)
        - _points_to_root_frame(ref_w, root_pos, root_quat),
        dim=-1,
    )
    swing = ~torch.as_tensor(manager.current_contact_state, dtype=torch.bool, device=ref_w.device)
    has_swing = swing.any(dim=-1)
    worst_error = torch.where(swing, foot_error, torch.full_like(foot_error, -torch.inf)).amax(dim=-1)
    reward = torch.exp(-torch.square(worst_error / float(std)))
    reward = torch.where(has_swing, reward, torch.zeros_like(reward))
    return reward * _parallelism_plan_valid(env, dtype=reward.dtype, device=reward.device)


def parallelism_tracking_errors(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> dict[str, torch.Tensor]:
    errors = update_parallelism_tracking_error_stats(env, asset_cfg)
    return {**errors, "joint_error": errors["joint_mean_error"]}
