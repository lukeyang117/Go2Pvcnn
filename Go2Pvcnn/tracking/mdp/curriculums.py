"""Curriculum terms for parallelism tracking."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tracking.mdp.rewards import (
    parallelism_tracking_episode_errors,
    reset_parallelism_tracking_error_stats,
)


def _lerp_range(initial: tuple[float, float], final: tuple[float, float], alpha: float) -> tuple[float, float]:
    return (
        float(initial[0]) + (float(final[0]) - float(initial[0])) * alpha,
        float(initial[1]) + (float(final[1]) - float(initial[1])) * alpha,
    )


def parallelism_velocity_curriculum(
    env,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    max_level: int = 10,
    lin_vel_threshold: float = 0.25,
    ang_vel_threshold: float = 0.35,
    joint_mean_threshold: float = 0.20,
    joint_max_threshold: float = 0.45,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    if not hasattr(command_term.cfg, "ranges") or not hasattr(command_term.cfg, "limit_ranges"):
        return torch.tensor(0.0, device=env.device)

    device = torch.device(getattr(env, "device", "cpu"))
    if not hasattr(env, "_parallelism_velocity_curriculum_level"):
        env._parallelism_velocity_curriculum_level = torch.zeros(int(env.num_envs), dtype=torch.long, device=device)

    ids = torch.as_tensor(env_ids, dtype=torch.long, device=device)
    errors = parallelism_tracking_episode_errors(env)
    time_out = getattr(env, "reset_time_outs", None)
    terminated = getattr(env, "reset_terminated", None)
    if time_out is None:
        time_out = torch.ones(int(env.num_envs), dtype=torch.bool, device=device)
    if terminated is None:
        terminated = torch.zeros(int(env.num_envs), dtype=torch.bool, device=device)
    success = (
        torch.as_tensor(time_out, dtype=torch.bool, device=device)[ids]
        & ~torch.as_tensor(terminated, dtype=torch.bool, device=device)[ids]
        & (errors["episode_lin_vel_error"][ids] < float(lin_vel_threshold))
        & (errors["episode_ang_vel_error"][ids] < float(ang_vel_threshold))
        & (errors["episode_joint_mean_error"][ids] < float(joint_mean_threshold))
        & (errors["episode_joint_max_error"][ids] < float(joint_max_threshold))
    )
    delta = torch.where(success, torch.ones_like(ids), -torch.ones_like(ids))
    levels = env._parallelism_velocity_curriculum_level
    levels[ids] = torch.clamp(levels[ids] + delta, 0, int(max_level))
    level_value = int(torch.round(torch.mean(levels.float())).item())
    alpha = float(level_value) / max(float(max_level), 1.0)

    initial_lin_x = (-0.1, 0.1)
    initial_lin_y = (-0.05, 0.05)
    initial_ang_z = (-0.2, 0.2)
    final = command_term.cfg.limit_ranges
    command_term.cfg.ranges.lin_vel_x = _lerp_range(initial_lin_x, final.lin_vel_x, alpha)
    command_term.cfg.ranges.lin_vel_y = _lerp_range(initial_lin_y, final.lin_vel_y, alpha)
    command_term.cfg.ranges.ang_vel_z = _lerp_range(initial_ang_z, final.ang_vel_z, alpha)
    reset_parallelism_tracking_error_stats(env, ids)
    return torch.mean(levels.float())
