"""Probe-only MPC debug variants exposed for viewer reproduction."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor

from .config import MpcPlannerCfg
from .kinematics import _JOINT_LIMITS, fk_feet_from_joint_angles, solve_joint_angles_from_trajectory
from .types import MpcPlannerTerrain, MpcRobotState


_REACHABLE_FK_VARIANTS = {
    "reachable_fk_cross_v1",
    "reachable_fk_cross_v2",
    "reachable_fk_cross_v3",
    "reachable_fk_cross_v4",
    "reachable_fk_cross_v5",
    "reachable_fk_cross_v6",
    "reachable_fk_cross_v7",
    "reachable_fk_cross_v8",
    "reachable_fk_cross_v9",
    "reachable_fk_cross_v10",
    "reachable_fk_cross_v11",
    "reachable_fk_cross_v12",
}


def apply_mpc_debug_variant_cfg(
    base_cfg: MpcPlannerCfg,
    variant: str | None,
    *,
    command: tuple[float, float, float] | Tensor | None = None,
) -> MpcPlannerCfg:
    """Return a copy of ``base_cfg`` with the requested V-series debug weights."""
    cfg = copy.deepcopy(base_cfg)
    variant_name = None if variant in (None, "", "baseline") else str(variant)
    cfg.debug_loss_variant = variant_name
    cfg.debug_loss_variant_cfg_applied = variant_name is not None
    if variant_name is None:
        return cfg

    def _lin_yaw(cmd) -> tuple[float | None, float]:
        if cmd is None:
            return None, 0.0
        cmd_t = torch.as_tensor(cmd, dtype=torch.float32).reshape(-1)
        vx = float(cmd_t[0].item()) if int(cmd_t.numel()) > 0 else 0.0
        vy = float(cmd_t[1].item()) if int(cmd_t.numel()) > 1 else 0.0
        yaw = float(cmd_t[2].item()) if int(cmd_t.numel()) > 2 else 0.0
        return math.hypot(vx, vy), abs(yaw)

    if variant_name == "reachable_loss_v1":
        cfg.losses.ik_fk_residual.weight *= 4.0
        cfg.losses.ik_fk_residual.contact_weight = max(float(cfg.losses.ik_fk_residual.contact_weight), 6.0)
        cfg.losses.kinematics.weight *= 4.0
        cfg.losses.kinematics.joint_limit_margin_rad = max(float(cfg.losses.kinematics.joint_limit_margin_rad), 0.20)
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 2.0
        cfg.losses.foot_trajectory_regularization.accel_weight *= 2.0
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.root_height.weight *= 1.5
        cfg.losses.support_plane_rp.weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 48)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 6.0e-3)
        return cfg
    if variant_name == "reachable_loss_small_v1":
        cfg = apply_mpc_debug_variant_cfg(cfg, "reachable_loss_v1", command=command)
        cfg.debug_loss_variant = variant_name
        cfg.losses.tracking.vel_weight *= 0.45
        cfg.losses.low_small_crossing.pass_margin_m = min(float(cfg.losses.low_small_crossing.pass_margin_m), 0.04)
        cfg.losses.low_small_crossing.obstacle_depth_m = min(float(cfg.losses.low_small_crossing.obstacle_depth_m), 0.16)
        cfg.losses.low_small_foot_over.radius_m = min(float(cfg.losses.low_small_foot_over.radius_m), 0.055)
        cfg.losses.low_small_foot_over.along_window_m = min(float(cfg.losses.low_small_foot_over.along_window_m), 0.20)
        cfg.losses.low_small_foot_over.clearance_m = min(float(cfg.losses.low_small_foot_over.clearance_m), 0.055)
        cfg.losses.low_small_foot_over.xy_weight *= 0.75
        cfg.losses.low_small_foot_over.direct_xy_weight *= 0.75
        cfg.losses.low_small_foot_over.z_weight *= 0.85
        cfg.losses.semantic_obstacle.soft_margin_m = min(float(cfg.losses.semantic_obstacle.soft_margin_m), 0.16)
        cfg.losses.semantic_contact_avoid.soft_margin_m = min(float(cfg.losses.semantic_contact_avoid.soft_margin_m), 0.14)
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if variant_name in {"reachable_fk_cross_v1", "reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
        cfg = apply_mpc_debug_variant_cfg(cfg, "reachable_loss_small_v1", command=command)
        cfg.debug_loss_variant = variant_name
        cfg.losses.tracking.vel_weight *= 0.40
        cfg.losses.progress.weight *= 1.35
        cfg.losses.low_small_crossing.weight *= 0.35
        cfg.losses.low_small_foot_over.xy_weight *= 0.55
        cfg.losses.low_small_foot_over.direct_xy_weight *= 0.55
        cfg.losses.low_small_foot_over.z_weight *= 0.75
        cfg.losses.ik_fk_residual.weight *= 2.0
        cfg.losses.kinematics.weight *= 2.0
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.25
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.35
        cfg.losses.low_small_stepcap.foot_boundary_weight *= 1.35
        cfg.losses.low_small_stepcap.foot_step_worst_weight *= 1.25
        cfg.losses.low_small_stepcap.foot_accel_weight *= 1.65
        cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.65
        cfg.losses.low_small_stepcap.foot_jerk_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 72)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 3.0e-3)
        if variant_name in {"reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
            cfg.losses.root_height.weight *= 3.0
            cfg.losses.support_plane_rp.weight *= 2.5
            cfg.losses.root_foot_center.weight *= 2.0
            cfg.losses.tracking.yaw_weight *= 1.25
            cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.75
            cfg.losses.low_small_stepcap.root_accel_weight *= 1.75
            cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.75
            cfg.losses.low_small_stepcap.foot_accel_weight *= 1.2
            cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.2
        if variant_name in {"reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
            cfg.losses.ik_fk_residual.weight *= 1.35
            cfg.losses.kinematics.weight *= 1.35
            cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.35
            cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.35
        if variant_name in {"reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
            cfg.losses.low_small_crossing.weight *= 0.65
            cfg.losses.low_small_foot_crossing.weight *= 1.4
            cfg.losses.semantic_contact_avoid.weight *= 1.25
            cfg.losses.low_small_stepcap.foot_step_worst_weight *= 1.2
            cfg.losses.low_small_stepcap.foot_accel_weight *= 1.15
            cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.15
        if variant_name == "reachable_fk_cross_v5":
            lin, yaw_abs = _lin_yaw(command)
            if lin is not None and lin <= 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.low_small_crossing.weight = 0.0
                cfg.losses.low_small_foot_over.weight = 0.0
                cfg.losses.low_small_foot_crossing.weight *= 1.25
                cfg.losses.low_small_stepcap.foot_step_worst_weight *= 1.10
                cfg.losses.low_small_stepcap.foot_accel_weight *= 1.10
                cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.10
            elif lin is not None and lin > 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.low_small_crossing.weight *= 0.45
                cfg.losses.low_small_foot_over.weight *= 0.45
                cfg.losses.progress.weight *= 1.35
                cfg.losses.tracking.vel_weight *= 0.75
                cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.25
                cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.25
        return cfg
    if variant_name == "reachable_fk_cross_v6":
        lin, yaw_abs = _lin_yaw(command)
        if lin is not None and lin <= 1.0e-4 and yaw_abs > 1.0e-4:
            cfg.losses.low_small_crossing.weight = 0.0
            cfg.losses.low_small_foot_over.weight = 0.0
            cfg.losses.low_small_foot_crossing.weight *= 1.25
            cfg.losses.semantic_contact_avoid.weight *= 1.20
            cfg.losses.ik_fk_residual.weight *= 1.10
            return cfg
        cfg = apply_mpc_debug_variant_cfg(cfg, "reachable_fk_cross_v4", command=command)
        cfg.debug_loss_variant = variant_name
        if lin is not None and lin > 1.0e-4 and yaw_abs > 1.0e-4:
            cfg.losses.low_small_crossing.weight *= 0.30
            cfg.losses.low_small_foot_over.weight *= 0.25
            cfg.losses.tracking.vel_weight = max(float(cfg.losses.tracking.vel_weight), 0.35)
            cfg.losses.tracking.yaw_weight *= 1.20
            cfg.losses.progress.weight *= 2.0
            cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.50
            cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.50
        return cfg
    if variant_name == "reachable_fk_cross_v7":
        cfg = apply_mpc_debug_variant_cfg(cfg, "reachable_fk_cross_v6", command=command)
        cfg.debug_loss_variant = variant_name
        lin, yaw_abs = _lin_yaw(command)
        if lin is not None and lin > 1.0e-4 and yaw_abs > 1.0e-4:
            cfg.losses.low_small_crossing.weight *= 0.40
            cfg.losses.low_small_foot_over.weight *= 0.40
            cfg.losses.tracking.vel_weight = max(float(cfg.losses.tracking.vel_weight), 0.50)
            cfg.losses.progress.min_progress_m = max(float(cfg.losses.progress.min_progress_m), 0.18)
            cfg.losses.progress.weight *= 1.50
        return cfg
    if variant_name == "reachable_fk_cross_v8":
        cfg = apply_mpc_debug_variant_cfg(cfg, "reachable_fk_cross_v7", command=command)
        cfg.debug_loss_variant = variant_name
        lin, yaw_abs = _lin_yaw(command)
        if lin is not None and lin > 1.0e-4 and yaw_abs > 1.0e-4:
            cfg.losses.root_height.weight *= 2.5
            cfg.losses.support_plane_rp.weight *= 1.5
            cfg.losses.root_foot_center.weight *= 1.5
            cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.25
        return cfg
    if variant_name in {"reachable_fk_cross_v9", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
        cfg = apply_mpc_debug_variant_cfg(cfg, "reachable_fk_cross_v8", command=command)
        cfg.debug_loss_variant = variant_name
        lin, yaw_abs = _lin_yaw(command)
        if lin is not None and lin > 1.0e-4 and yaw_abs > 1.0e-4:
            if variant_name == "reachable_fk_cross_v12":
                cfg.losses.ik_fk_residual.weight *= 1.20
                cfg.losses.kinematics.weight *= 1.15
            else:
                cfg.losses.ik_fk_residual.weight *= 1.55 if variant_name == "reachable_fk_cross_v11" else 1.75
                cfg.losses.kinematics.weight *= 1.35 if variant_name == "reachable_fk_cross_v11" else 1.50
            cfg.losses.kinematics.joint_limit_margin_rad = max(float(cfg.losses.kinematics.joint_limit_margin_rad), 0.32)
            if variant_name in {"reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
                cfg.losses.low_small_crossing.weight *= 0.85
                cfg.losses.low_small_foot_over.weight *= 0.85
                cfg.losses.foot_trajectory_regularization.accel_weight *= 1.25
                cfg.losses.low_small_stepcap.foot_accel_weight *= 1.25
                cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.25
            if variant_name == "reachable_fk_cross_v12":
                cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 96)
                cfg.runtime.lr = min(float(cfg.runtime.lr), 2.5e-3)
        return cfg
    raise ValueError(f"Unknown MPC debug variant {variant_name!r}")


def reachable_distance_window_weights(
    root_xy: Tensor,
    obstacle_xy: Tensor,
    *,
    command: Tensor,
    min_cross_distance_m: float = 0.14,
    max_cross_distance_m: float = 0.28,
    sigma_m: float = 0.05,
) -> dict[str, Tensor]:
    root = torch.as_tensor(root_xy)
    dtype = root.dtype
    device = root.device
    obs = torch.as_tensor(obstacle_xy, dtype=dtype, device=device)
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0).expand(int(root.shape[0]), -1)
    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    heading = cmd[:, :2] / speed.clamp_min(1.0e-6).unsqueeze(-1)
    root0 = root[:, 0]
    root_end = root[:, -1]
    start_distance = ((obs - root0) * heading).sum(dim=-1)
    end_distance = ((obs - root_end) * heading).sum(dim=-1)
    lo = float(min_cross_distance_m)
    hi = float(max_cross_distance_m)
    center = 0.5 * (lo + hi)
    half_width = max(0.5 * (hi - lo), 1.0e-6)
    sigma = max(float(sigma_m), 1.0e-6)
    cross_weight = torch.exp(-0.5 * ((start_distance - center) / sigma).square())
    cross_weight = cross_weight * torch.sigmoid((start_distance - lo) / sigma) * torch.sigmoid((hi - start_distance) / sigma)
    cross_weight = torch.where(speed > 1.0e-6, cross_weight.clamp(0.0, 1.0), torch.zeros_like(cross_weight))
    approach_weight = torch.sigmoid((start_distance - hi) / sigma)
    approach_weight = torch.where(speed > 1.0e-6, approach_weight.clamp(0.0, 1.0), torch.zeros_like(approach_weight))
    too_close_weight = torch.sigmoid((lo - start_distance) / sigma)
    return {
        "approach_weight": approach_weight,
        "cross_weight": cross_weight,
        "too_close_weight": too_close_weight,
        "start_distance": start_distance,
        "end_distance": end_distance,
        "target_distance": torch.full_like(start_distance, center).clamp(lo, hi),
        "window_half_width": torch.full_like(start_distance, half_width),
    }


def _finite_horizon_touchdown_phase(swing_center: Tensor, swing_width: Tensor) -> Tensor:
    return torch.clamp(swing_center + 0.5 * swing_width, min=0.0, max=1.0)


def _sample_time_noncyclic(values: Tensor, phase: Tensor) -> Tensor:
    batch, horizon, legs, *tail = values.shape
    pos = torch.clamp(phase, 0.0, 1.0) * float(max(horizon - 1, 1))
    i0 = torch.floor(pos).to(dtype=torch.long).clamp(0, horizon - 1)
    i1 = (i0 + 1).clamp(0, horizon - 1)
    alpha = (pos - torch.floor(pos)).to(dtype=values.dtype)
    b = torch.arange(batch, device=values.device).view(batch, 1).expand(batch, legs)
    l = torch.arange(legs, device=values.device).view(1, legs).expand(batch, legs)
    v0 = values[b, i0, l]
    v1 = values[b, i1, l]
    return torch.lerp(v0, v1, alpha.view(batch, legs, *([1] * len(tail))))


def _low_small_obstacle_for_debug_variant(
    terrain: MpcPlannerTerrain,
    state: MpcRobotState,
    command: Tensor,
    cfg: MpcPlannerCfg,
) -> tuple[Tensor | None, Tensor | None]:
    if terrain.semantic_map is None:
        return None, None
    from .semantic_policy import classify_semantic_obstacle_mode

    policy = classify_semantic_obstacle_mode(terrain, state, command, cfg)
    batch = int(torch.as_tensor(state.root_pos).shape[0])
    dtype = torch.as_tensor(state.root_pos).dtype
    device = torch.as_tensor(state.root_pos).device
    height = torch.as_tensor(terrain.height_map, dtype=dtype, device=device)
    if height.ndim == 2:
        height = height.unsqueeze(0)
    if int(height.shape[0]) == 1 and batch > 1:
        height = height.expand(batch, -1, -1)
    semantic = torch.as_tensor(terrain.semantic_map, dtype=torch.long, device=device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0)
    if int(semantic.shape[0]) == 1 and batch > 1:
        semantic = semantic.expand(batch, -1, -1)
    if int(height.shape[0]) != batch or int(semantic.shape[0]) != batch:
        return policy.obstacle_xy, torch.full((batch,), 0.16, dtype=dtype, device=device)
    grid_sem = semantic.reshape(batch, -1)
    from .losses.terrain_clearance import _nearby_height_for_sparse_semantic, _semantic_id_mask, _terrain_grid_world_xy

    grid_z = _nearby_height_for_sparse_semantic(terrain, height, dtype=dtype, device=device)
    root_ground = torch.as_tensor(state.root_pos, dtype=dtype, device=device)[:, 2]
    small = _semantic_id_mask(grid_sem, cfg.losses.touchdown_semantic.small_ids)
    low_small = torch.logical_and(
        small,
        (grid_z - root_ground[:, None]) <= float(cfg.losses.low_small_crossing.high_small_relative_height_m),
    )
    grid_xy = _terrain_grid_world_xy(terrain, dtype=dtype, device=device)
    obs_xy = policy.obstacle_xy
    dist = torch.linalg.vector_norm(grid_xy - obs_xy[:, None, :], dim=-1)
    masked_dist = torch.where(low_small, dist, torch.full_like(dist, 1.0e6))
    close = masked_dist <= 0.18
    obstacle_top = torch.where(close, grid_z, torch.full_like(grid_z, -1.0e6)).amax(dim=-1)
    fallback_top = torch.full_like(obstacle_top, 0.16)
    obstacle_top = torch.where(torch.isfinite(obstacle_top) & (obstacle_top > -1.0e5), obstacle_top, fallback_top)
    return obs_xy, obstacle_top


def mpc_debug_extra_loss(
    decoded: object,
    *,
    variant: str | None,
    command: Tensor,
    obstacle_xy: Tensor | None = None,
    obstacle_height: Tensor | float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return V-series extra loss terms shared by probes and viewer debug runs."""
    root = torch.as_tensor(decoded.root_pos)
    variant_name = None if variant in (None, "", "baseline") else str(variant)
    if variant_name not in _REACHABLE_FK_VARIANTS:
        zero = torch.zeros((int(root.shape[0]),), dtype=root.dtype, device=root.device)
        return zero, {}
    if command is None or obstacle_xy is None:
        zero = torch.zeros((int(root.shape[0]),), dtype=root.dtype, device=root.device)
        return zero, {}

    rpy = torch.as_tensor(decoded.root_rpy, dtype=root.dtype, device=root.device)
    foot = torch.as_tensor(decoded.foot_pos, dtype=root.dtype, device=root.device)
    contact_prob = torch.as_tensor(decoded.contact_prob, dtype=root.dtype, device=root.device)
    joint = solve_joint_angles_from_trajectory(root, rpy, foot, clamp_to_limits=True)
    fk_foot = fk_feet_from_joint_angles(root, rpy, joint)
    residual = torch.linalg.vector_norm(fk_foot - foot, dim=-1)
    raw_joint = solve_joint_angles_from_trajectory(root, rpy, foot, clamp_to_limits=False)
    limits = _JOINT_LIMITS.to(device=root.device, dtype=root.dtype).view(1, 1, 12, 2)
    raw_limit_excess = torch.relu(limits[..., 0] - raw_joint).square() + torch.relu(raw_joint - limits[..., 1]).square()
    contact_mass = torch.clamp(contact_prob.sum(dim=(1, 2)), min=1.0)
    residual_contact = (contact_prob * residual).sum(dim=(1, 2)) / contact_mass
    residual_loss = residual.mean(dim=(1, 2)) + 4.0 * residual_contact
    if int(fk_foot.shape[1]) >= 2:
        step = torch.linalg.vector_norm(fk_foot[:, 1:] - fk_foot[:, :-1], dim=-1)
        step_loss = step.square().mean(dim=(1, 2)) + 8.0 * step.amax(dim=(1, 2)).square()
    else:
        step_loss = torch.zeros_like(residual_loss)
    if int(fk_foot.shape[1]) >= 3:
        accel = torch.linalg.vector_norm(fk_foot[:, 2:] - 2.0 * fk_foot[:, 1:-1] + fk_foot[:, :-2], dim=-1)
        accel_loss = accel.square().mean(dim=(1, 2)) + 12.0 * accel.amax(dim=(1, 2)).square()
    else:
        accel_loss = torch.zeros_like(residual_loss)

    batch = int(root.shape[0])
    cmd = torch.as_tensor(command, dtype=root.dtype, device=root.device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=root.dtype, device=root.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    obs = torch.as_tensor(obstacle_xy, dtype=root.dtype, device=root.device)
    if obs.ndim == 1:
        obs = obs.unsqueeze(0).expand(batch, -1)
    height_t = torch.as_tensor(0.16 if obstacle_height is None else obstacle_height, dtype=root.dtype, device=root.device)
    if height_t.ndim == 0:
        height_t = height_t.expand(batch)

    weights = reachable_distance_window_weights(root[..., :2], obs, command=cmd)
    speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    translation_active = speed > 1.0e-4
    heading = cmd[:, :2] / speed.clamp_min(1.0e-6).unsqueeze(-1)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    rel = fk_foot[..., :2] - obs[:, None, None, :]
    along = (rel * heading[:, None, None, :]).sum(dim=-1)
    lateral = (rel * left[:, None, None, :]).sum(dim=-1)
    lane = torch.exp(-0.5 * (lateral / 0.075).square())
    swing_weight = (1.0 - contact_prob).clamp_min(0.0)
    near_obs = torch.exp(-0.5 * (along / 0.13).square())
    clearance_target = height_t[:, None, None] + 0.055
    lift_deficit = torch.relu(clearance_target - fk_foot[..., 2]).square()
    root_disp = root[:, -1, :2] - root[:, 0, :2]
    lateral_drift = torch.abs((root_disp * left).sum(dim=-1))
    root_height_min = root[..., 2].amin(dim=1)
    path_rel = root[..., :2] - root[:, :1, :2]
    path_lateral = torch.abs((path_rel * left[:, None, :]).sum(dim=-1))
    path_lateral_max = path_lateral.amax(dim=1)
    valid_posture = torch.ones_like(weights["cross_weight"])
    if variant_name in {"reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
        height_gate = torch.sigmoid((root_height_min - 0.12) / 0.02)
        drift_signal = path_lateral_max if variant_name in {"reachable_fk_cross_v3", "reachable_fk_cross_v4"} else lateral_drift
        drift_gate = torch.sigmoid((0.14 - drift_signal) / 0.04)
        valid_posture = height_gate * drift_gate
    if variant_name in {"reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"} and not bool(torch.any(translation_active).item()):
        breakdown = {
            "reachable_yaw_only_reachability": 80.0 * residual_loss
            + 90.0 * (raw_limit_excess.mean(dim=(1, 2)) + raw_limit_excess.amax(dim=(1, 2))),
            "reachable_yaw_only_fk_step": 2.0 * step_loss,
            "reachable_yaw_only_fk_accel": 3.0 * accel_loss,
        }
        total = sum(breakdown.values())
        return torch.nan_to_num(total, nan=1.0e6, posinf=1.0e6, neginf=1.0e6), {
            name: torch.nan_to_num(value, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
            for name, value in breakdown.items()
        }

    cross_gate = weights["cross_weight"][:, None, None] * valid_posture[:, None, None] * swing_weight * lane * near_obs
    cross_mass = cross_gate.sum(dim=(1, 2)).clamp_min(1.0)
    cross_over = (cross_gate * lift_deficit).sum(dim=(1, 2)) / cross_mass
    missing_cross = weights["cross_weight"] * torch.relu(0.35 - cross_gate.sum(dim=(1, 2))).square()
    end_distance = weights["end_distance"]
    target_distance = weights["target_distance"]
    approach_distance = weights["approach_weight"] * torch.relu(end_distance - target_distance).square()
    too_close = weights["too_close_weight"] * torch.relu(target_distance - end_distance).square() * 0.25
    breakdown = {
        "reachable_fk_residual": 95.0 * residual_loss,
        "reachable_fk_worst_residual": 160.0 * (residual.amax(dim=(1, 2)).square() + torch.relu(residual - 0.08).square().mean(dim=(1, 2))),
        "reachable_raw_joint_limit_excess": 220.0 * (raw_limit_excess.mean(dim=(1, 2)) + raw_limit_excess.amax(dim=(1, 2))),
        "reachable_fk_step": 12.0 * step_loss,
        "reachable_fk_accel": 16.0 * accel_loss,
        "reachable_fk_cross_window": 55.0 * missing_cross,
        "reachable_fk_cross_over": 180.0 * cross_over,
        "reachable_fk_approach_distance": 90.0 * approach_distance + 15.0 * too_close,
        "reachable_fk_direction_lateral": 20.0 * weights["approach_weight"] * lateral_drift.square(),
    }
    if variant_name in {"reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
        foot_y = fk_foot[..., 1]
        foot_spread = foot_y.amax(dim=-1) - foot_y.amin(dim=-1)
        breakdown.update(
            {
                "reachable_fk_base_height_guard": 420.0 * torch.relu(0.12 - root_height_min).square(),
                "reachable_fk_cross_posture_gate": 75.0 * weights["cross_weight"] * torch.relu(0.55 - valid_posture).square(),
                "reachable_fk_direction_lateral": 120.0 * (weights["approach_weight"] + weights["cross_weight"]) * torch.relu(lateral_drift - 0.10).square(),
                "reachable_fk_spider_guard": 45.0 * torch.relu(foot_spread.amax(dim=1) - 0.72).square(),
            }
        )
    if variant_name in {"reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
        path_gate = weights["approach_weight"] + weights["cross_weight"]
        lateral_cap = 0.05 if variant_name in {"reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"} else (0.06 if variant_name == "reachable_fk_cross_v4" else 0.08)
        direction_cap = 0.035 if variant_name in {"reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"} else (0.045 if variant_name == "reachable_fk_cross_v4" else 0.06)
        breakdown.update(
            {
                "reachable_fk_lateral_path_guard": 360.0 * path_gate * torch.relu(path_lateral_max - lateral_cap).square(),
                "reachable_fk_direction_lateral": 320.0 * path_gate * torch.relu(path_lateral_max - direction_cap).square(),
                "reachable_fk_residual": 125.0 * residual_loss,
                "reachable_fk_worst_residual": 240.0 * (
                    residual.amax(dim=(1, 2)).square() + torch.relu(residual - 0.06).square().mean(dim=(1, 2))
                ),
            }
        )
    if variant_name in {"reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
        lane_contact = torch.exp(-0.5 * (lateral / 0.06).square()) * torch.exp(-0.5 * (along / 0.10).square())
        contact_or_low = torch.maximum(contact_prob, torch.relu((height_t[:, None, None] + 0.02) - fk_foot[..., 2]) / 0.05)
        small_contact = (lane_contact * contact_or_low.clamp(0.0, 1.0)).amax(dim=(1, 2)).square()
        path_gate = weights["approach_weight"] + weights["cross_weight"]
        breakdown.update(
            {
                "reachable_fk_small_contact_guard": 620.0 * path_gate * small_contact,
                "reachable_fk_lateral_path_guard": 700.0 * path_gate * torch.relu(path_lateral_max - 0.045).square(),
                "reachable_fk_direction_lateral": 620.0 * path_gate * torch.relu(path_lateral_max - 0.035).square(),
            }
        )
    if variant_name in {"reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
        dxy = root[:, -1, :2] - root[:, 0, :2]
        along_progress = (dxy * heading).sum(dim=-1)
        disp_norm = torch.linalg.vector_norm(dxy, dim=-1).clamp_min(1.0e-6)
        direction_cos = along_progress / disp_norm
        desired_progress = 0.35 * speed
        mixed_gate = torch.logical_and(translation_active, torch.abs(cmd[:, 2]) > 1.0e-4).to(dtype=root.dtype, device=root.device)
        breakdown.update(
            {
                "reachable_fk_command_direction_cosine": 420.0 * mixed_gate * torch.relu(0.65 - direction_cos).square(),
                "reachable_fk_command_progress": 180.0 * mixed_gate * torch.relu(desired_progress - along_progress).square(),
            }
        )
        if variant_name in {"reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
            breakdown.update(
                {
                    "reachable_fk_mixed_base_height_guard": 850.0 * mixed_gate * torch.relu(0.14 - root_height_min).square(),
                    "reachable_fk_cross_posture_gate": 180.0 * mixed_gate * torch.relu(0.75 - valid_posture).square(),
                }
            )
        if variant_name in {"reachable_fk_cross_v9", "reachable_fk_cross_v10", "reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
            mixed_reach = (
                residual.amax(dim=(1, 2)).square()
                + torch.relu(residual - 0.05).square().mean(dim=(1, 2))
                + raw_limit_excess.amax(dim=(1, 2))
            )
            breakdown.update(
                {
                    "reachable_fk_mixed_reachability_barrier": 420.0 * mixed_gate * mixed_reach,
                    "reachable_fk_residual": 180.0 * residual_loss,
                    "reachable_fk_worst_residual": 360.0 * (
                        residual.amax(dim=(1, 2)).square()
                        + torch.relu(residual - 0.05).square().mean(dim=(1, 2))
                    ),
                }
            )
        if variant_name in {"reachable_fk_cross_v11", "reachable_fk_cross_v12"}:
            touchdown_phase = _finite_horizon_touchdown_phase(decoded.swing_center, decoded.swing_width)
            touchdown = _sample_time_noncyclic(foot, touchdown_phase)
            touchdown_root = _sample_time_noncyclic(root.unsqueeze(2).expand(-1, -1, 4, -1), touchdown_phase)
            touchdown_rpy = _sample_time_noncyclic(rpy.unsqueeze(2).expand(-1, -1, 4, -1), touchdown_phase)
            touchdown_targets = touchdown[:, None, :, :].expand(batch, 4, 4, 3)
            td_joint = solve_joint_angles_from_trajectory(
                touchdown_root,
                touchdown_rpy,
                touchdown_targets,
                clamp_to_limits=True,
            )
            td_fk_all = fk_feet_from_joint_angles(touchdown_root, touchdown_rpy, td_joint)
            leg_ids = torch.arange(4, device=root.device)
            td_fk = td_fk_all[:, leg_ids, leg_ids]
            td_residual = torch.linalg.vector_norm(td_fk - touchdown, dim=-1)
            foot_along = ((foot[..., :2] - obs[:, None, None, :]) * heading[:, None, None, :]).sum(dim=-1)
            touchdown_along = ((touchdown[..., :2] - obs[:, None, :]) * heading[:, None, :]).sum(dim=-1)
            swing_active = swing_weight > 0.25
            swing_forward_extreme = torch.where(
                swing_active,
                foot_along,
                torch.full_like(foot_along, -1.0e6),
            ).amax(dim=1)
            endpoint_tolerance = 0.0 if variant_name == "reachable_fk_cross_v12" else 0.035
            endpoint_backtrack = torch.relu(swing_forward_extreme - touchdown_along - endpoint_tolerance)
            root_z = root[..., 2].unsqueeze(-1)
            above_root = torch.relu(foot[..., 2] - root_z - 0.025)
            v11_gate = torch.logical_and(translation_active, torch.abs(cmd[:, 2]) > 1.0e-4).to(dtype=root.dtype, device=root.device)
            path_gate = torch.clamp(weights["approach_weight"] + weights["cross_weight"], 0.0, 1.0)
            active_gate = torch.maximum(v11_gate, path_gate)
            endpoint_weight = 20000.0 if variant_name == "reachable_fk_cross_v12" else 520.0
            reach_weight = 120.0 if variant_name == "reachable_fk_cross_v12" else 380.0
            height_weight = 80.0 if variant_name == "reachable_fk_cross_v12" else 260.0
            endpoint_loss = endpoint_backtrack.square().amax(dim=1)
            if variant_name == "reachable_fk_cross_v12":
                endpoint_loss = endpoint_loss + 0.35 * endpoint_backtrack.mean(dim=1) + 0.65 * endpoint_backtrack.amax(dim=1)
            breakdown.update(
                {
                    "reachable_fk_touchdown_endpoint_consistency": endpoint_weight * active_gate * endpoint_loss,
                    "reachable_fk_sampled_touchdown_reachability": reach_weight * active_gate * (
                        td_residual.amax(dim=1).square() + torch.relu(td_residual - 0.05).square().mean(dim=1)
                    ),
                    "reachable_fk_foot_above_root_guard": height_weight * active_gate * (
                        (swing_weight * above_root.square()).sum(dim=(1, 2)) / swing_weight.sum(dim=(1, 2)).clamp_min(1.0)
                        + above_root.amax(dim=(1, 2)).square()
                    ),
                }
            )
    total = sum(breakdown.values())
    total = torch.nan_to_num(total, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    return total, {name: torch.nan_to_num(value, nan=1.0e6, posinf=1.0e6, neginf=1.0e6) for name, value in breakdown.items()}


def mpc_debug_extra_loss_from_terrain(
    decoded: object,
    state: MpcRobotState,
    command: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcPlannerCfg,
) -> tuple[Tensor, dict[str, Tensor]]:
    variant = cfg.debug_loss_variant
    if variant in (None, "", "baseline"):
        zero = torch.zeros((int(decoded.root_pos.shape[0]),), dtype=decoded.root_pos.dtype, device=decoded.root_pos.device)
        return zero, {}
    obstacle_xy, obstacle_height = _low_small_obstacle_for_debug_variant(terrain, state, command, cfg)
    return mpc_debug_extra_loss(
        decoded,
        variant=variant,
        command=command,
        obstacle_xy=obstacle_xy,
        obstacle_height=obstacle_height,
    )


__all__ = [
    "apply_mpc_debug_variant_cfg",
    "mpc_debug_extra_loss",
    "mpc_debug_extra_loss_from_terrain",
    "reachable_distance_window_weights",
]
