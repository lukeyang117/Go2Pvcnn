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


def reference_joint_pos_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.35,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).current_joint_pos
    actual = torch.as_tensor(asset.data.joint_pos, dtype=ref.dtype, device=ref.device)
    error = torch.mean(torch.square(actual - ref), dim=-1)
    return torch.exp(-error / (float(std) * float(std)))


def reference_joint_vel_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 2.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).current_joint_vel
    actual = torch.as_tensor(asset.data.joint_vel, dtype=ref.dtype, device=ref.device)
    error = torch.mean(torch.square(actual - ref), dim=-1)
    return torch.exp(-error / (float(std) * float(std)))


def parallelism_tracking_errors(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> dict[str, torch.Tensor]:
    asset = env.scene[asset_cfg.name]
    manager = get_parallelism_reference_manager(env)
    command = env.command_manager.get_command("base_velocity")
    command = torch.as_tensor(command, dtype=asset.data.root_lin_vel_b.dtype, device=asset.data.root_lin_vel_b.device)
    lin_vel_error = torch.linalg.vector_norm(asset.data.root_lin_vel_b[:, :2] - command[:, :2], dim=-1)
    ang_vel_error = torch.abs(asset.data.root_ang_vel_b[:, 2] - command[:, 2])
    joint_error = torch.mean(torch.abs(asset.data.joint_pos - manager.current_joint_pos), dim=-1)
    return {
        "lin_vel_error": lin_vel_error,
        "ang_vel_error": ang_vel_error,
        "joint_error": joint_error,
    }
