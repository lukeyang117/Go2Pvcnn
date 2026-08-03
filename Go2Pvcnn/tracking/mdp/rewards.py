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


def _current_parallelism_tracking_errors(env, asset_cfg: SceneEntityCfg) -> dict[str, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    manager = get_parallelism_reference_manager(env)
    ref_lin = manager.current_root_lin_vel_b_policy
    ref_ang = manager.current_root_ang_vel_b_policy
    actual_lin = torch.as_tensor(asset.data.root_lin_vel_b, dtype=ref_lin.dtype, device=ref_lin.device)
    actual_ang = torch.as_tensor(asset.data.root_ang_vel_b, dtype=ref_ang.dtype, device=ref_ang.device)
    ref_joint = manager.current_joint_pos
    actual_joint = torch.as_tensor(asset.data.joint_pos, dtype=ref_joint.dtype, device=ref_joint.device)
    joint_abs_error = torch.abs(actual_joint - ref_joint)
    return {
        "lin_vel_error": torch.linalg.vector_norm(actual_lin - ref_lin, dim=-1),
        "ang_vel_error": torch.linalg.vector_norm(actual_ang - ref_ang, dim=-1),
        "joint_mean_error": torch.mean(joint_abs_error, dim=-1),
        "joint_max_error": torch.max(joint_abs_error, dim=-1).values,
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
        "_parallelism_tracking_lin_vel_sum",
        "_parallelism_tracking_ang_vel_sum",
        "_parallelism_tracking_error_frames",
    ):
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
    return {
        "episode_joint_mean_error": env._parallelism_tracking_joint_mean_sum / frames,
        "episode_joint_max_error": env._parallelism_tracking_joint_max,
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
    ref = get_parallelism_reference_manager(env).current_joint_pos
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
    ref = get_parallelism_reference_manager(env).current_joint_vel
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
    ref = get_parallelism_reference_manager(env).current_root_lin_vel_b_policy
    actual = torch.as_tensor(asset.data.root_lin_vel_b, dtype=ref.dtype, device=ref.device)
    update_parallelism_tracking_error_stats(env, asset_cfg)
    return _gaussian_error_reward(actual - ref, std)


def reference_root_ang_vel_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).current_root_ang_vel_b_policy
    actual = torch.as_tensor(asset.data.root_ang_vel_b, dtype=ref.dtype, device=ref.device)
    update_parallelism_tracking_error_stats(env, asset_cfg)
    return _gaussian_error_reward(actual - ref, std)


def parallelism_tracking_errors(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> dict[str, torch.Tensor]:
    errors = update_parallelism_tracking_error_stats(env, asset_cfg)
    return {
        "lin_vel_error": errors["lin_vel_error"],
        "ang_vel_error": errors["ang_vel_error"],
        "joint_error": errors["joint_mean_error"],
        "joint_mean_error": errors["joint_mean_error"],
        "joint_max_error": errors["joint_max_error"],
        "episode_lin_vel_error": errors["episode_lin_vel_error"],
        "episode_ang_vel_error": errors["episode_ang_vel_error"],
        "episode_joint_mean_error": errors["episode_joint_mean_error"],
        "episode_joint_max_error": errors["episode_joint_max_error"],
    }
