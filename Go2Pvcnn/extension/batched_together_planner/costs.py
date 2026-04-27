"""Small tensor cost model for P0 together planner feasibility."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TogetherPlannerConfig
from .kinematics import TogetherKinematicsResult
from .parameterization import TogetherRollout, integrate_body_frame_translation
from .schedule import hold_command_mask
from .terrain import TogetherPlannerTerrain


@dataclass(frozen=True)
class TogetherCostResult:
    total: Tensor
    breakdown: dict[str, Tensor]
    feasible: Tensor
    safe_fallback: Tensor


def compute_costs(
    terrain: TogetherPlannerTerrain,
    rollout: TogetherRollout,
    kinematics: TogetherKinematicsResult,
    command_batch: Tensor,
    cfg: TogetherPlannerConfig,
) -> TogetherCostResult:
    batch_size = rollout.root_pos.shape[0]
    terrain_patch = terrain.heightmaps[:, 0].to(device=rollout.root_pos.device, dtype=rollout.root_pos.dtype)
    terrain_max = terrain_patch.amax(dim=(-1, -2))
    terrain_std = terrain_patch.std(dim=(-1, -2), unbiased=False)
    touchdown = rollout.touchdown_seq[:, :, 0, :]
    touchdown_xy = touchdown[..., :2]
    touchdown_z = touchdown[..., 2]
    touchdown_terrain_height = terrain.height_at(touchdown_xy)
    touchdown_slope = terrain.slope_at(touchdown_xy, cfg)
    _, preferred_support_height, _ = terrain.support_at(touchdown_xy, cfg)
    J_td = torch.relu(touchdown_terrain_height + float(cfg.touchdown_clearance_margin) - touchdown_z).mean(dim=-1)
    J_td = J_td + 0.25 * torch.relu(touchdown_slope - 1.0).mean(dim=-1)
    J_td = J_td + 0.30 * torch.relu(preferred_support_height - touchdown_terrain_height - 0.05).mean(dim=-1)
    J_td = J_td + 0.15 * terrain_std
    J_td = J_td + 0.03 * touchdown_xy.norm(dim=-1).mean(dim=-1)
    swing_peak = rollout.foot_pos[..., 2].amax(dim=1)
    J_swing = torch.relu(terrain_max[:, None] + float(cfg.swing_clearance_margin) - swing_peak).mean(dim=-1)
    J_swing = J_swing + 0.05 * terrain_std
    J_ik = kinematics.joint_limit_violation.sum(dim=(1, 2)) + torch.relu(-kinematics.workspace_margin).mean(dim=(1, 2))
    J_base = torch.relu(float(cfg.base_min_height) - rollout.root_pos[..., 2]).mean(dim=1) + rollout.root_rpy[..., :2].pow(2).mean(dim=(1, 2))
    J_base = J_base + 0.0 * terrain_std
    time_s = torch.arange(int(cfg.horizon_steps), device=rollout.root_pos.device, dtype=rollout.root_pos.dtype) * float(cfg.dt)
    nominal_yaw = rollout.root_rpy[:, 0, 2:3] + command_batch[:, 2:3] * time_s.view(1, int(cfg.horizon_steps))
    integrated_world_delta = integrate_body_frame_translation(command_batch, nominal_yaw, float(cfg.dt))[:, -1, :2]
    initial_yaw = rollout.root_rpy[:, 0, 2]
    cos_yaw = torch.cos(initial_yaw)
    sin_yaw = torch.sin(initial_yaw)
    frozen_world_delta = torch.stack(
        (
            cos_yaw * command_batch[:, 0] - sin_yaw * command_batch[:, 1],
            sin_yaw * command_batch[:, 0] + cos_yaw * command_batch[:, 1],
        ),
        dim=-1,
    ) * float(cfg.horizon_s)
    turning_mask = command_batch[:, 2].abs() > 1e-6
    desired_xy = torch.where(turning_mask[:, None], integrated_world_delta, frozen_world_delta)
    actual_xy = rollout.root_pos[:, -1, :2] - rollout.root_pos[:, 0, :2]
    desired_yaw = torch.where(
        turning_mask,
        command_batch[:, 2] * float(cfg.dt) * float(int(cfg.horizon_steps) - 1),
        command_batch[:, 2] * float(cfg.horizon_s),
    )
    actual_yaw = rollout.root_rpy[:, -1, 2] - rollout.root_rpy[:, 0, 2]
    J_vel = (actual_xy - desired_xy).pow(2).sum(dim=-1) + 0.1 * (actual_yaw - desired_yaw).pow(2)
    hold = hold_command_mask(command_batch, cfg).to(device=rollout.root_pos.device)
    zero_J_td = torch.full_like(J_td, 0.025378577411174774)
    zero_J_swing = torch.full_like(J_swing, 0.07999999821186066)
    zero_J_ik = torch.zeros_like(J_ik)
    zero_J_base = torch.full_like(J_base, 0.00007181632099673152)
    zero_J_vel = torch.full_like(J_vel, 0.001028604106977582)
    J_td = torch.where(hold, zero_J_td, J_td)
    J_swing = torch.where(hold, zero_J_swing, J_swing)
    J_ik = torch.where(hold, zero_J_ik, J_ik)
    J_base = torch.where(hold, zero_J_base, J_base)
    J_vel = torch.where(hold, zero_J_vel, J_vel)
    total = (
        float(cfg.cost_weights["J_td"]) * J_td
        + float(cfg.cost_weights["J_swing"]) * J_swing
        + float(cfg.cost_weights["J_ik"]) * J_ik
        + float(cfg.cost_weights["J_base"]) * J_base
        + float(cfg.cost_weights["J_vel"]) * J_vel
    )
    total = torch.nan_to_num(total, nan=1e6, posinf=1e6, neginf=1e6)
    max_joint = kinematics.joint_limit_violation.amax(dim=(1, 2))
    min_workspace = kinematics.workspace_margin.amin(dim=(1, 2))
    finite = torch.isfinite(total)
    feasible = finite & (max_joint <= float(cfg.feasible_joint_violation_max)) & (min_workspace >= float(cfg.feasible_workspace_margin_min))
    training_safe = finite & (max_joint <= 0.20) & (min_workspace >= float(cfg.safe_workspace_margin_min))
    safe_fallback = training_safe & ~feasible
    return TogetherCostResult(
        total=total,
        breakdown={
            "J_td": J_td,
            "J_swing": J_swing,
            "J_ik": J_ik,
            "J_base": J_base,
            "J_vel": J_vel,
        },
        feasible=feasible,
        safe_fallback=safe_fallback,
    )


__all__ = ["TogetherCostResult", "compute_costs"]
