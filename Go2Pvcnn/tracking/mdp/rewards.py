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


def _current_parallelism_tracking_errors(env, asset_cfg: SceneEntityCfg) -> dict[str, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    manager = get_parallelism_reference_manager(env)
    ref_lin = manager.step_root_lin_vel_b_policy
    ref_ang = manager.step_root_ang_vel_b_policy
    actual_lin = torch.as_tensor(asset.data.root_lin_vel_b, dtype=ref_lin.dtype, device=ref_lin.device)
    actual_ang = torch.as_tensor(asset.data.root_ang_vel_b, dtype=ref_ang.dtype, device=ref_ang.device)
    ref_joint = manager.step_joint_pos
    actual_joint = torch.as_tensor(asset.data.joint_pos, dtype=ref_joint.dtype, device=ref_joint.device)
    joint_abs_error = torch.abs(actual_joint - ref_joint)
    if hasattr(asset.data, "body_pos_w"):
        ref_foot = manager.step_foot_pos_w
        actual_foot = _actual_foot_pos_w(
            asset,
            SceneEntityCfg(asset_cfg.name, body_names=".*_foot"),
            ref_foot,
        )
        root_pos = torch.as_tensor(asset.data.root_pos_w, dtype=ref_foot.dtype, device=ref_foot.device)
        root_quat = torch.as_tensor(asset.data.root_quat_w, dtype=ref_foot.dtype, device=ref_foot.device)
        foot_abs_error = torch.linalg.vector_norm(
            _points_to_root_frame(actual_foot, root_pos, root_quat)
            - _points_to_root_frame(ref_foot, root_pos, root_quat),
            dim=-1,
        )
    else:
        foot_abs_error = torch.zeros(
            actual_joint.shape[0],
            4,
            dtype=actual_joint.dtype,
            device=actual_joint.device,
        )
    return {
        "lin_vel_error": torch.linalg.vector_norm(actual_lin - ref_lin, dim=-1),
        "ang_vel_error": torch.linalg.vector_norm(actual_ang - ref_ang, dim=-1),
        "joint_mean_error": torch.mean(joint_abs_error, dim=-1),
        "joint_max_error": torch.max(joint_abs_error, dim=-1).values,
        "foot_mean_error": torch.mean(foot_abs_error, dim=-1),
        "foot_max_error": torch.max(foot_abs_error, dim=-1).values,
    }


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

    device = errors["lin_vel_error"].device
    count = int(getattr(env, "num_envs", errors["lin_vel_error"].shape[0]))
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

    if not hasattr(env, "_parallelism_tracking_joint_mean_sum"):
        env._parallelism_tracking_joint_mean_sum = torch.zeros(count, dtype=errors["joint_mean_error"].dtype, device=device)
        env._parallelism_tracking_joint_max = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_foot_mean_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_foot_max = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_lin_vel_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_ang_vel_sum = torch.zeros_like(env._parallelism_tracking_joint_mean_sum)
        env._parallelism_tracking_error_frames = torch.zeros(count, dtype=torch.long, device=device)

    env._parallelism_tracking_joint_mean_sum += torch.where(
        update_mask, errors["joint_mean_error"], torch.zeros_like(errors["joint_mean_error"])
    )
    env._parallelism_tracking_joint_max = torch.maximum(
        env._parallelism_tracking_joint_max,
        torch.where(update_mask, errors["joint_max_error"], torch.zeros_like(errors["joint_max_error"])),
    )
    env._parallelism_tracking_foot_mean_sum += torch.where(
        update_mask, errors["foot_mean_error"], torch.zeros_like(errors["foot_mean_error"])
    )
    env._parallelism_tracking_foot_max = torch.maximum(
        env._parallelism_tracking_foot_max,
        torch.where(update_mask, errors["foot_max_error"], torch.zeros_like(errors["foot_max_error"])),
    )
    env._parallelism_tracking_lin_vel_sum += torch.where(
        update_mask, errors["lin_vel_error"], torch.zeros_like(errors["lin_vel_error"])
    )
    env._parallelism_tracking_ang_vel_sum += torch.where(
        update_mask, errors["ang_vel_error"], torch.zeros_like(errors["ang_vel_error"])
    )
    env._parallelism_tracking_error_frames += update_mask.to(dtype=torch.long)
    env._parallelism_tracking_last_step = torch.where(update_mask, step, last_step)
    frames = env._parallelism_tracking_error_frames.clamp_min(1).to(dtype=errors["joint_mean_error"].dtype)
    return {
        **errors,
        "episode_joint_mean_error": env._parallelism_tracking_joint_mean_sum / frames,
        "episode_joint_max_error": env._parallelism_tracking_joint_max,
        "episode_foot_mean_error": env._parallelism_tracking_foot_mean_sum / frames,
        "episode_foot_max_error": env._parallelism_tracking_foot_max,
        "episode_lin_vel_error": env._parallelism_tracking_lin_vel_sum / frames,
        "episode_ang_vel_error": env._parallelism_tracking_ang_vel_sum / frames,
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
        "_parallelism_tracking_lin_vel_sum",
        "_parallelism_tracking_ang_vel_sum",
        "_parallelism_tracking_error_frames",
    ):
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
    frames = env._parallelism_tracking_error_frames.clamp_min(1).to(
        dtype=env._parallelism_tracking_joint_mean_sum.dtype
    )
    foot_mean_sum = getattr(env, "_parallelism_tracking_foot_mean_sum", torch.zeros_like(env._parallelism_tracking_joint_mean_sum))
    foot_max = getattr(env, "_parallelism_tracking_foot_max", torch.zeros_like(env._parallelism_tracking_joint_mean_sum))
    return {
        "episode_joint_mean_error": env._parallelism_tracking_joint_mean_sum / frames,
        "episode_joint_max_error": env._parallelism_tracking_joint_max,
        "episode_foot_mean_error": foot_mean_sum / frames,
        "episode_foot_max_error": foot_max,
        "episode_lin_vel_error": env._parallelism_tracking_lin_vel_sum / frames,
        "episode_ang_vel_error": env._parallelism_tracking_ang_vel_sum / frames,
    }


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
    return _gaussian_error_reward(error, std)


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
    return _gaussian_error_reward(error, std)


def reference_root_lin_vel_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).step_root_lin_vel_b_policy
    actual = torch.as_tensor(asset.data.root_lin_vel_b, dtype=ref.dtype, device=ref.device)
    update_parallelism_tracking_error_stats(env, asset_cfg)
    return _gaussian_error_reward(actual - ref, std)


def reference_root_ang_vel_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).step_root_ang_vel_b_policy
    actual = torch.as_tensor(asset.data.root_ang_vel_b, dtype=ref.dtype, device=ref.device)
    update_parallelism_tracking_error_stats(env, asset_cfg)
    return _gaussian_error_reward(actual - ref, std)


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
    actual_w = _actual_foot_pos_w(asset, asset_cfg, ref_w)
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
    return torch.exp(-normalized_error / (float(std) * float(std)))


def parallelism_tracking_errors(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> dict[str, torch.Tensor]:
    errors = update_parallelism_tracking_error_stats(env, asset_cfg)
    return {
        "lin_vel_error": errors["lin_vel_error"],
        "ang_vel_error": errors["ang_vel_error"],
        "joint_error": errors["joint_mean_error"],
        "joint_mean_error": errors["joint_mean_error"],
        "joint_max_error": errors["joint_max_error"],
        "foot_mean_error": errors["foot_mean_error"],
        "foot_max_error": errors["foot_max_error"],
        "episode_lin_vel_error": errors["episode_lin_vel_error"],
        "episode_ang_vel_error": errors["episode_ang_vel_error"],
        "episode_joint_mean_error": errors["episode_joint_mean_error"],
        "episode_joint_max_error": errors["episode_joint_max_error"],
        "episode_foot_mean_error": errors["episode_foot_mean_error"],
        "episode_foot_max_error": errors["episode_foot_max_error"],
    }
