from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT, GO2PVCNN_ROOT / "tests"):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles, fk_leg_points_from_joint_angles, solve_joint_angles_from_trajectory  # noqa: E402
from extension.batch_mpc_planner.losses.terrain_clearance import finite_horizon_touchdown_phase, sample_time  # noqa: E402
from extension.batch_mpc_planner.planner import plan_segment, sample_touchdown_positions  # noqa: E402
from extension.batch_mpc_planner.semantic_policy import shape_nominal_command_for_semantic_obstacles  # noqa: E402
from mpc_semantic_obstacle_jitter_probe import (  # noqa: E402
    _candidate_variants_for_variant,
    _command_heading_yaw,
    _command_relative_xy,
    _parse_command,
    _plan_rolling_viewer_trajectory,
    _root_rpy_from_viewer_result,
    _semantic_probe_seed,
    _set_env0_yaw,
    _variant_cfg,
)
from fixtures.viewer_runtime_diagnostics import RealViewerRuntimeFixture, refresh_targeted_scanner_pose  # noqa: E402


_JOINT_LIMITS = torch.tensor(
    (
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
    ),
    dtype=torch.float32,
)


DEFAULT_COMMANDS = (
    "forward_v050:0.50 0.00 0.00",
    "lateral_v050:0.00 0.50 0.00",
    "diagonal_v050:0.35 0.35 0.00",
    "mixed_yaw_v050:0.50 0.25 1.00",
    "yaw100:0.00 0.00 1.00",
)

PARAMETRIC_VARIANTS = {"parametric_v1"}


def _removed_debug_variant_error(variant: str) -> RuntimeError:
    return RuntimeError(
        f"MPC debug variant {variant!r} was removed from extension.batch_mpc_planner. "
        "Probe-only variants must live under Go2Pvcnn/tests, not in the production planner package."
    )


def apply_mpc_debug_variant_cfg(base_cfg, variant_name: str | None, command=None):
    del base_cfg, command
    raise _removed_debug_variant_error(str(variant_name))


def mpc_debug_extra_loss(*args, **kwargs):
    variant = str(kwargs.get("variant", "unknown"))
    raise _removed_debug_variant_error(variant)


def _command_frame(command: tuple[float, float, float], *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, float]:
    cmd_xy = torch.tensor(command[:2], dtype=dtype, device=device)
    norm = torch.linalg.vector_norm(cmd_xy)
    if float(norm.item()) <= 1.0e-6:
        forward = torch.tensor((1.0, 0.0), dtype=dtype, device=device)
        return forward, torch.tensor((-0.0, 1.0), dtype=dtype, device=device), 0.0
    forward = cmd_xy / norm
    lateral = torch.stack((-forward[1], forward[0]))
    return forward, lateral, float(norm.item())


def reachable_cfg_for_variant(base_cfg, variant: str, command: tuple[float, float, float] | None = None):
    if str(variant) in PARAMETRIC_VARIANTS:
        cfg = copy.deepcopy(base_cfg)
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 40)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-2)
        return cfg
    if str(variant) in {
        "reachable_fk_cross_v1",
        "reachable_fk_cross_v2",
        "reachable_fk_cross_v3",
        "reachable_fk_cross_v4",
        "reachable_fk_cross_v5",
        "reachable_fk_cross_v6",
        "reachable_fk_cross_v7",
        "reachable_fk_cross_v8",
        "reachable_fk_cross_v9",
        "reachable_fk_cross_v11",
        "reachable_fk_cross_v12",
    }:
        return apply_mpc_debug_variant_cfg(base_cfg, str(variant), command=command)
    cfg = copy.deepcopy(base_cfg)
    if str(variant) == "baseline":
        return cfg
    if str(variant) == "reachable_loss_v1":
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
    if str(variant) == "reachable_loss_v2":
        cfg = reachable_cfg_for_variant(cfg, "reachable_loss_v1")
        cfg.losses.tracking.vel_weight *= 0.35
        cfg.losses.low_small_crossing.weight *= 0.35
        cfg.losses.low_small_foot_crossing.weight *= 0.70
        cfg.losses.progress.weight *= 1.5
        cfg.losses.ik_fk_residual.weight *= 1.75
        cfg.losses.ik_fk_residual.contact_weight = max(float(cfg.losses.ik_fk_residual.contact_weight), 10.0)
        cfg.losses.kinematics.weight *= 1.75
        cfg.losses.kinematics.joint_limit_margin_rad = max(float(cfg.losses.kinematics.joint_limit_margin_rad), 0.28)
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.75
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.75
        cfg.losses.smoothness.foot_weight *= 1.5
        cfg.losses.low_small_stepcap.foot_boundary_weight *= 1.75
        cfg.losses.low_small_stepcap.foot_step_worst_weight *= 1.75
        cfg.losses.low_small_stepcap.foot_accel_weight *= 2.0
        cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 2.0
        cfg.losses.low_small_stepcap.foot_jerk_weight *= 1.75
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        return cfg
    if str(variant) == "reachable_struct_v1":
        cfg = reachable_cfg_for_variant(cfg, "reachable_loss_v1")
        cfg.losses.tracking.vel_weight *= 0.60
        cfg.losses.low_small_crossing.weight *= 0.60
        cfg.losses.low_small_stepcap.foot_accel_weight *= 1.5
        cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.5
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 56)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 5.0e-3)
        return cfg
    if str(variant) in {"reachable_struct_v2", "reachable_struct_v3"}:
        cfg = reachable_cfg_for_variant(cfg, "reachable_loss_v1")
        cfg.losses.tracking.vel_weight *= 0.50
        cfg.losses.low_small_crossing.weight *= 0.50
        cfg.losses.low_small_foot_crossing.weight *= 0.80
        cfg.losses.ik_fk_residual.weight *= 2.0
        cfg.losses.ik_fk_residual.contact_weight = max(float(cfg.losses.ik_fk_residual.contact_weight), 12.0)
        cfg.losses.kinematics.weight *= 2.0
        cfg.losses.kinematics.joint_limit_margin_rad = max(float(cfg.losses.kinematics.joint_limit_margin_rad), 0.30)
        cfg.losses.low_small_stepcap.foot_boundary_weight *= 1.25
        cfg.losses.low_small_stepcap.foot_accel_weight *= 1.25
        cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 4.0e-3)
        if str(variant) == "reachable_struct_v3":
            cfg.losses.tracking.vel_weight *= 0.80
            cfg.losses.ik_fk_residual.weight *= 1.5
            cfg.losses.kinematics.weight *= 1.5
            cfg.losses.foot_trajectory_regularization.boundary_weight *= 0.75
            cfg.losses.foot_trajectory_regularization.accel_weight *= 0.75
            cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 72)
            cfg.runtime.lr = min(float(cfg.runtime.lr), 3.0e-3)
        return cfg
    if str(variant) == "reachable_loss_small_v1":
        cfg = reachable_cfg_for_variant(cfg, "reachable_loss_v1")
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
    if str(variant) == "reachable_loss_small_v2":
        cfg = reachable_cfg_for_variant(cfg, "reachable_loss_small_v1")
        cfg.losses.foot_trajectory_regularization.boundary_weight *= 1.35
        cfg.losses.foot_trajectory_regularization.accel_weight *= 1.60
        cfg.losses.low_small_stepcap.foot_boundary_weight *= 1.45
        cfg.losses.low_small_stepcap.foot_step_worst_weight *= 1.35
        cfg.losses.low_small_stepcap.foot_accel_weight *= 2.0
        cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 2.0
        cfg.losses.low_small_stepcap.foot_jerk_weight *= 1.5
        cfg.losses.smoothness.foot_weight *= 1.25
        cfg.runtime.optimize_steps = max(int(cfg.runtime.optimize_steps), 64)
        cfg.runtime.lr = min(float(cfg.runtime.lr), 3.5e-3)
        return cfg
    if str(variant) == "reachable_fk_cross_v6":
        lin = None if command is None else math.hypot(float(command[0]), float(command[1]))
        yaw_abs = 0.0 if command is None else abs(float(command[2]))
        if lin is not None and lin <= 1.0e-4 and yaw_abs > 1.0e-4:
            cfg.losses.low_small_crossing.weight = 0.0
            cfg.losses.low_small_foot_over.weight = 0.0
            cfg.losses.low_small_foot_crossing.weight *= 1.25
            cfg.losses.semantic_contact_avoid.weight *= 1.20
            cfg.losses.ik_fk_residual.weight *= 1.10
            return cfg
        cfg = reachable_cfg_for_variant(cfg, "reachable_fk_cross_v4")
        if lin is not None and lin > 1.0e-4 and yaw_abs > 1.0e-4:
            cfg.losses.low_small_crossing.weight *= 0.30
            cfg.losses.low_small_foot_over.weight *= 0.25
            cfg.losses.tracking.vel_weight = max(float(cfg.losses.tracking.vel_weight), 0.35)
            cfg.losses.tracking.yaw_weight *= 1.20
            cfg.losses.progress.weight *= 2.0
            cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.50
            cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.50
        return cfg
    if str(variant) == "reachable_fk_cross_v7":
        cfg = reachable_cfg_for_variant(cfg, "reachable_fk_cross_v6", command=command)
        if command is not None:
            lin = math.hypot(float(command[0]), float(command[1]))
            yaw_abs = abs(float(command[2]))
            if lin > 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.low_small_crossing.weight *= 0.40
                cfg.losses.low_small_foot_over.weight *= 0.40
                cfg.losses.tracking.vel_weight = max(float(cfg.losses.tracking.vel_weight), 0.50)
                cfg.losses.progress.min_progress_m = max(float(cfg.losses.progress.min_progress_m), 0.18)
                cfg.losses.progress.weight *= 1.50
        return cfg
    if str(variant) == "reachable_fk_cross_v8":
        cfg = reachable_cfg_for_variant(cfg, "reachable_fk_cross_v7", command=command)
        if command is not None:
            lin = math.hypot(float(command[0]), float(command[1]))
            yaw_abs = abs(float(command[2]))
            if lin > 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.root_height.weight *= 2.5
                cfg.losses.support_plane_rp.weight *= 1.5
                cfg.losses.root_foot_center.weight *= 1.5
                cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.25
        return cfg
    if str(variant) == "reachable_fk_cross_v9":
        cfg = reachable_cfg_for_variant(cfg, "reachable_fk_cross_v8", command=command)
        if command is not None:
            lin = math.hypot(float(command[0]), float(command[1]))
            yaw_abs = abs(float(command[2]))
            if lin > 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.ik_fk_residual.weight *= 1.75
                cfg.losses.kinematics.weight *= 1.50
                cfg.losses.kinematics.joint_limit_margin_rad = max(float(cfg.losses.kinematics.joint_limit_margin_rad), 0.32)
        return cfg
    if str(variant) == "reachable_fk_cross_v10":
        cfg = reachable_cfg_for_variant(cfg, "reachable_fk_cross_v7", command=command)
        if command is not None:
            lin = math.hypot(float(command[0]), float(command[1]))
            yaw_abs = abs(float(command[2]))
            if lin > 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.root_height.weight *= 1.45
                cfg.losses.support_plane_rp.weight *= 1.20
                cfg.losses.ik_fk_residual.weight *= 1.35
                cfg.losses.kinematics.weight *= 1.20
                cfg.losses.low_small_crossing.weight *= 0.75
                cfg.losses.low_small_foot_over.weight *= 0.75
                cfg.losses.progress.min_progress_m = max(float(cfg.losses.progress.min_progress_m), 0.14)
        return cfg
    if str(variant) in {"reachable_fk_cross_v1", "reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
        cfg = reachable_cfg_for_variant(cfg, "reachable_loss_small_v1")
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
        if str(variant) in {"reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
            cfg.losses.root_height.weight *= 3.0
            cfg.losses.support_plane_rp.weight *= 2.5
            cfg.losses.root_foot_center.weight *= 2.0
            cfg.losses.tracking.yaw_weight *= 1.25
            cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.75
            cfg.losses.low_small_stepcap.root_accel_weight *= 1.75
            cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.75
            cfg.losses.low_small_stepcap.foot_accel_weight *= 1.2
            cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.2
        if str(variant) in {"reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
            cfg.losses.ik_fk_residual.weight *= 1.35
            cfg.losses.kinematics.weight *= 1.35
            cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.35
            cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.35
        if str(variant) in {"reachable_fk_cross_v4", "reachable_fk_cross_v5"}:
            cfg.losses.low_small_crossing.weight *= 0.65
            cfg.losses.low_small_foot_crossing.weight *= 1.4
            cfg.losses.semantic_contact_avoid.weight *= 1.25
            cfg.losses.low_small_stepcap.foot_step_worst_weight *= 1.2
            cfg.losses.low_small_stepcap.foot_accel_weight *= 1.15
            cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.15
        if str(variant) == "reachable_fk_cross_v5" and command is not None:
            lin = math.hypot(float(command[0]), float(command[1]))
            yaw_abs = abs(float(command[2]))
            if lin <= 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.low_small_crossing.weight = 0.0
                cfg.losses.low_small_foot_over.weight = 0.0
                cfg.losses.low_small_foot_crossing.weight *= 1.25
                cfg.losses.low_small_stepcap.foot_step_worst_weight *= 1.10
                cfg.losses.low_small_stepcap.foot_accel_weight *= 1.10
                cfg.losses.low_small_stepcap.foot_accel_worst_weight *= 1.10
            elif lin > 1.0e-4 and yaw_abs > 1.0e-4:
                cfg.losses.low_small_crossing.weight *= 0.45
                cfg.losses.low_small_foot_over.weight *= 0.45
                cfg.losses.progress.weight *= 1.35
                cfg.losses.tracking.vel_weight *= 0.75
                cfg.losses.low_small_stepcap.root_step_worst_weight *= 1.25
                cfg.losses.low_small_stepcap.root_accel_worst_weight *= 1.25
        return cfg
    raise ValueError(f"Unknown reachable crossing variant {variant!r}")


def reachable_distance_window_weights(
    root_xy: torch.Tensor,
    obstacle_xy: torch.Tensor,
    *,
    command: torch.Tensor,
    min_cross_distance_m: float = 0.14,
    max_cross_distance_m: float = 0.28,
    sigma_m: float = 0.05,
) -> dict[str, torch.Tensor]:
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


def reachable_extra_loss(
    decoded,
    *,
    variant: str,
    command: torch.Tensor | None = None,
    obstacle_xy: torch.Tensor | None = None,
    obstacle_height: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    root = torch.as_tensor(decoded.root_pos)
    if str(variant) not in {
        "reachable_struct_v1",
        "reachable_struct_v2",
        "reachable_struct_v3",
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
    }:
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
    raw_lower_excess = torch.relu(limits[..., 0] - raw_joint)
    raw_upper_excess = torch.relu(raw_joint - limits[..., 1])
    raw_limit_excess = raw_lower_excess.square() + raw_upper_excess.square()
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
    if str(variant) in {
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
    }:
        if str(variant) != "reachable_fk_cross_v10":
            return mpc_debug_extra_loss(
                decoded,
                variant=str(variant),
                command=command,
                obstacle_xy=obstacle_xy,
                obstacle_height=obstacle_height,
            )
        batch = int(root.shape[0])
        if command is None or obstacle_xy is None:
            zero = torch.zeros((batch,), dtype=root.dtype, device=root.device)
            return zero, {}
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
        valid_posture = torch.ones_like(weights["cross_weight"])
        path_rel = root[..., :2] - root[:, :1, :2]
        path_lateral = torch.abs((path_rel * left[:, None, :]).sum(dim=-1))
        path_lateral_max = path_lateral.amax(dim=1)
        if str(variant) in {"reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"}:
            height_gate = torch.sigmoid((root_height_min - 0.12) / 0.02)
            drift_signal = path_lateral_max if str(variant) in {"reachable_fk_cross_v3", "reachable_fk_cross_v4"} else lateral_drift
            drift_gate = torch.sigmoid((0.14 - drift_signal) / 0.04)
            valid_posture = height_gate * drift_gate
        if str(variant) in {"reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"} and not bool(torch.any(translation_active).item()):
            breakdown = {
                "reachable_yaw_only_reachability": 80.0 * residual_loss
                + 90.0 * (raw_limit_excess.mean(dim=(1, 2)) + raw_limit_excess.amax(dim=(1, 2))),
                "reachable_yaw_only_fk_step": 2.0 * step_loss,
                "reachable_yaw_only_fk_accel": 3.0 * accel_loss,
            }
            total = sum(breakdown.values())
            total = torch.nan_to_num(total, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
            return total, {
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
        lateral_drift_loss = lateral_drift.square()
        breakdown = {
            "reachable_fk_residual": 95.0 * residual_loss,
            "reachable_fk_worst_residual": 160.0 * (residual.amax(dim=(1, 2)).square() + torch.relu(residual - 0.08).square().mean(dim=(1, 2))),
            "reachable_raw_joint_limit_excess": 220.0 * (raw_limit_excess.mean(dim=(1, 2)) + raw_limit_excess.amax(dim=(1, 2))),
            "reachable_fk_step": 12.0 * step_loss,
            "reachable_fk_accel": 16.0 * accel_loss,
            "reachable_fk_cross_window": 55.0 * missing_cross,
            "reachable_fk_cross_over": 180.0 * cross_over,
            "reachable_fk_approach_distance": 90.0 * approach_distance + 15.0 * too_close,
            "reachable_fk_direction_lateral": 20.0 * weights["approach_weight"] * lateral_drift_loss,
        }
        if str(variant) in {"reachable_fk_cross_v2", "reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"}:
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
        if str(variant) in {"reachable_fk_cross_v3", "reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"}:
            path_gate = weights["approach_weight"] + weights["cross_weight"]
            lateral_cap = 0.05 if str(variant) in {"reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"} else (0.06 if str(variant) == "reachable_fk_cross_v4" else 0.08)
            direction_cap = 0.035 if str(variant) in {"reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"} else (0.045 if str(variant) == "reachable_fk_cross_v4" else 0.06)
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
        if str(variant) in {"reachable_fk_cross_v4", "reachable_fk_cross_v5", "reachable_fk_cross_v6", "reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"}:
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
        if str(variant) in {"reachable_fk_cross_v7", "reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"}:
            dxy = root[:, -1, :2] - root[:, 0, :2]
            along_progress = (dxy * heading).sum(dim=-1)
            disp_norm = torch.linalg.vector_norm(dxy, dim=-1).clamp_min(1.0e-6)
            direction_cos = along_progress / disp_norm
            desired_progress = 0.35 * speed
            yaw_active = torch.abs(cmd[:, 2]) > 1.0e-4
            mixed_gate = torch.logical_and(translation_active, yaw_active).to(dtype=root.dtype, device=root.device)
            breakdown.update(
                {
                    "reachable_fk_command_direction_cosine": 420.0 * mixed_gate * torch.relu(0.65 - direction_cos).square(),
                    "reachable_fk_command_progress": 180.0 * mixed_gate * torch.relu(desired_progress - along_progress).square(),
                }
            )
            if str(variant) in {"reachable_fk_cross_v8", "reachable_fk_cross_v9", "reachable_fk_cross_v10"}:
                base_guard_weight = 850.0 if str(variant) in {"reachable_fk_cross_v8", "reachable_fk_cross_v9"} else 520.0
                posture_weight = 180.0 if str(variant) in {"reachable_fk_cross_v8", "reachable_fk_cross_v9"} else 90.0
                breakdown.update(
                    {
                        "reachable_fk_mixed_base_height_guard": base_guard_weight * mixed_gate * torch.relu(0.14 - root_height_min).square(),
                        "reachable_fk_cross_posture_gate": posture_weight * mixed_gate * torch.relu(0.75 - valid_posture).square(),
                    }
                )
            if str(variant) in {"reachable_fk_cross_v9", "reachable_fk_cross_v10"}:
                mixed_reach = (
                    residual.amax(dim=(1, 2)).square()
                    + torch.relu(residual - 0.05).square().mean(dim=(1, 2))
                    + raw_limit_excess.amax(dim=(1, 2))
                )
                reach_weight = 420.0 if str(variant) == "reachable_fk_cross_v9" else 230.0
                residual_weight = 180.0 if str(variant) == "reachable_fk_cross_v9" else 145.0
                worst_weight = 360.0 if str(variant) == "reachable_fk_cross_v9" else 260.0
                breakdown.update(
                    {
                        "reachable_fk_mixed_reachability_barrier": reach_weight * mixed_gate * mixed_reach,
                        "reachable_fk_residual": residual_weight * residual_loss,
                        "reachable_fk_worst_residual": worst_weight * (
                            residual.amax(dim=(1, 2)).square()
                            + torch.relu(residual - 0.05).square().mean(dim=(1, 2))
                        ),
                    }
                )
            if str(variant) == "reachable_fk_cross_v10":
                balance = (
                    torch.relu(0.72 - direction_cos).square()
                    + torch.relu(0.125 - root_height_min).square()
                    + torch.relu(residual.amax(dim=(1, 2)) - 0.32).square()
                    + torch.relu(path_lateral_max - 0.22).square()
                )
                breakdown.update({"reachable_fk_mixed_soft_balance": 150.0 * mixed_gate * balance})
    elif str(variant) == "reachable_struct_v1":
        breakdown = {
            "reachable_fk_residual": 80.0 * residual_loss,
            "reachable_fk_step": 40.0 * step_loss,
            "reachable_fk_accel": 50.0 * accel_loss,
        }
    else:
        touchdown_phase = finite_horizon_touchdown_phase(
            torch.as_tensor(decoded.swing_center, dtype=root.dtype, device=root.device),
            torch.as_tensor(decoded.swing_width, dtype=root.dtype, device=root.device),
        )
        touchdown_residual = sample_time(residual.unsqueeze(-1), touchdown_phase, cyclic=False).squeeze(-1)
        touchdown_contact = torch.clamp(contact_prob.max(dim=1).values, min=0.25)
        touchdown_mass = torch.clamp(touchdown_contact.sum(dim=1), min=1.0)
        touchdown_loss = touchdown_residual.mean(dim=1) + 5.0 * (touchdown_contact * touchdown_residual).sum(dim=1) / touchdown_mass
        if str(variant) == "reachable_struct_v2":
            breakdown = {
                "reachable_fk_residual": 100.0 * residual_loss,
                "reachable_touchdown_fk_residual": 140.0 * touchdown_loss,
                "reachable_fk_step": 18.0 * step_loss,
                "reachable_fk_accel": 22.0 * accel_loss,
            }
        else:
            worst_residual = residual.amax(dim=(1, 2)).square()
            residual_barrier = torch.relu(residual - 0.08).square().mean(dim=(1, 2))
            raw_limit_loss = raw_limit_excess.mean(dim=(1, 2)) + 2.0 * raw_limit_excess.amax(dim=(1, 2))
            breakdown = {
                "reachable_fk_residual": 90.0 * residual_loss,
                "reachable_fk_worst_residual": 220.0 * (worst_residual + residual_barrier),
                "reachable_raw_joint_limit_excess": 260.0 * raw_limit_loss,
                "reachable_touchdown_fk_residual": 120.0 * touchdown_loss,
                "reachable_fk_step": 10.0 * step_loss,
                "reachable_fk_accel": 12.0 * accel_loss,
            }
    total = sum(breakdown.values())
    total = torch.nan_to_num(total, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
    return total, {name: torch.nan_to_num(value, nan=1.0e6, posinf=1.0e6, neginf=1.0e6) for name, value in breakdown.items()}


@contextmanager
def _patched_reachable_loss_for_variant(
    variant: str,
    *,
    command: torch.Tensor | None = None,
    obstacle_xy: torch.Tensor | None = None,
    obstacle_height: float | None = None,
):
    if str(variant) not in {
        "reachable_struct_v1",
        "reachable_struct_v2",
        "reachable_struct_v3",
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
    }:
        yield
        return
    del command, obstacle_xy, obstacle_height
    yield


def _project_command_frame(xy: torch.Tensor, origin_xy: torch.Tensor, command: tuple[float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
    xy_t = torch.as_tensor(xy, dtype=torch.float32)
    origin = torch.as_tensor(origin_xy, dtype=torch.float32, device=xy_t.device)
    forward, lateral, _ = _command_frame(command, device=xy_t.device, dtype=xy_t.dtype)
    rel = xy_t - origin
    along = (rel * forward).sum(dim=-1)
    lateral_coord = (rel * lateral).sum(dim=-1)
    return along, lateral_coord


def _touchdown_summary(
    touchdown: torch.Tensor,
    obstacle_xy: torch.Tensor,
    command: tuple[float, float, float],
) -> dict[str, list[float]]:
    td = torch.as_tensor(touchdown, dtype=torch.float32)
    if td.ndim == 4:
        td = td[:, 0]
    obstacle = torch.as_tensor(obstacle_xy, dtype=torch.float32, device=td.device)
    along, lateral = _project_command_frame(td[..., :2], obstacle, command)
    return {
        "along_m": [float(v) for v in along.reshape(-1).detach().cpu().tolist()],
        "lateral_m": [float(v) for v in lateral.reshape(-1).detach().cpu().tolist()],
        "z_m": [float(v) for v in td[..., 2].reshape(-1).detach().cpu().tolist()],
    }


def _touchdown_delta_summary(
    before: torch.Tensor,
    after: torch.Tensor,
    obstacle_xy: torch.Tensor,
    command: tuple[float, float, float],
) -> dict[str, list[float] | float]:
    before_t = torch.as_tensor(before, dtype=torch.float32)
    after_t = torch.as_tensor(after, dtype=torch.float32, device=before_t.device)
    if before_t.ndim == 4:
        before_t = before_t[:, 0]
    if after_t.ndim == 4:
        after_t = after_t[:, 0]
    before_along, _ = _project_command_frame(before_t[..., :2], obstacle_xy, command)
    after_along, _ = _project_command_frame(after_t[..., :2], obstacle_xy, command)
    delta = after_t - before_t
    delta_along = after_along - before_along
    return {
        "delta_along_m": [float(v) for v in delta_along.reshape(-1).detach().cpu().tolist()],
        "delta_z_m": [float(v) for v in delta[..., 2].reshape(-1).detach().cpu().tolist()],
        "delta_xyz_norm_max_m": float(torch.linalg.vector_norm(delta, dim=-1).max().item()),
    }


def reachable_touchdown_chain_trace(
    terrain,
    state,
    command_tensor: torch.Tensor,
    cfg,
    obstacle_xy: torch.Tensor,
    command: tuple[float, float, float],
) -> dict[str, object]:
    planning_command, _ = shape_nominal_command_for_semantic_obstacles(terrain, state, command_tensor, cfg)
    result = plan_segment(terrain, state, planning_command, cfg=cfg)
    exported_touchdown = result.planned_touchdown_w[:, 0]
    root = torch.as_tensor(result.root_pos, dtype=torch.float32)
    rpy = torch.as_tensor(result.root_rpy, dtype=torch.float32, device=root.device)
    clamped_joint = solve_joint_angles_from_trajectory(
        root,
        rpy,
        torch.as_tensor(result.foot_pos, dtype=root.dtype, device=root.device),
        clamp_to_limits=True,
    )
    fk_foot = fk_feet_from_joint_angles(root, rpy, clamped_joint)
    target = exported_touchdown.to(dtype=fk_foot.dtype, device=fk_foot.device)
    dist = torch.linalg.vector_norm(fk_foot - target[:, None, :, :], dim=-1)
    frame_idx = torch.argmin(dist, dim=1)
    leg_idx = torch.arange(4, device=fk_foot.device).view(1, 4).expand(int(fk_foot.shape[0]), 4)
    batch_idx = torch.arange(int(fk_foot.shape[0]), device=fk_foot.device).view(-1, 1).expand_as(leg_idx)
    fk_touchdown = fk_foot[batch_idx, frame_idx, leg_idx]
    exported = _touchdown_summary(exported_touchdown, obstacle_xy, command)
    fk = _touchdown_summary(fk_touchdown, obstacle_xy, command)
    return {
        "type": "reachable_touchdown_chain_trace",
        "optimized_export": exported,
        "fk_from_clamped_ik": fk,
        "delta_optimized_export_to_fk": _touchdown_delta_summary(
            exported_touchdown,
            fk_touchdown,
            obstacle_xy,
            command,
        ),
    }


def reachable_command_direction_metrics(root_pos: torch.Tensor, *, command: tuple[float, float, float]) -> dict[str, float | int]:
    root = torch.as_tensor(root_pos, dtype=torch.float32)
    forward, lateral, translation_norm = _command_frame(command, device=root.device, dtype=root.dtype)
    displacement = root[:, -1, :2] - root[:, 0, :2]
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
    along = (displacement * forward).sum(dim=-1)
    lateral_drift = torch.abs((displacement * lateral).sum(dim=-1))
    active = int(translation_norm > 1.0e-6)
    cosine = torch.zeros_like(along)
    if active:
        cosine = along / displacement_norm.clamp_min(1.0e-6)
    duration_s = max(int(root.shape[1]) - 1, 1) * 0.02
    expected_progress = float(translation_norm) * float(duration_s)
    return {
        "translation_command_active": active,
        "command_direction_cosine": float(cosine.mean().item()),
        "along_progress_m": float(along.mean().item()),
        "lateral_drift_m": float(lateral_drift.max().item()),
        "speed_magnitude_tracking_error": float(torch.abs(along - expected_progress).mean().item()) if active else 0.0,
    }


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor, *, default: float = 0.0) -> float:
    num = float(torch.as_tensor(numerator, dtype=torch.float64).item())
    den = float(torch.as_tensor(denominator, dtype=torch.float64).item())
    if abs(den) <= 1.0e-9:
        return float(default)
    return num / den


def reachable_swing_continuity_metrics(
    fk_foot_pos: torch.Tensor,
    contact_state: torch.Tensor,
    *,
    replan_interval_steps: int,
) -> dict[str, float]:
    foot = torch.as_tensor(fk_foot_pos, dtype=torch.float32)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    if int(foot.shape[1]) < 2:
        return {
            "fk_swing_foot_step_max_to_median": 0.0,
            "fk_swing_foot_accel_max_to_mean": 0.0,
            "replan_boundary_fk_foot_step_to_median": 0.0,
        }
    swing = ~contact
    step = torch.linalg.vector_norm(foot[:, 1:] - foot[:, :-1], dim=-1)
    swing_step_mask = torch.logical_or(swing[:, 1:], swing[:, :-1])
    swing_steps = step[swing_step_mask]
    if int(swing_steps.numel()) == 0:
        swing_steps = step.reshape(-1)
    median_step = torch.median(swing_steps).clamp_min(1.0e-9)
    step_ratio = _safe_ratio(swing_steps.max(), median_step)
    if int(foot.shape[1]) >= 3:
        accel = torch.linalg.vector_norm(foot[:, 2:] - 2.0 * foot[:, 1:-1] + foot[:, :-2], dim=-1)
        swing_accel_mask = torch.logical_or(torch.logical_or(swing[:, 2:], swing[:, 1:-1]), swing[:, :-2])
        swing_accels = accel[swing_accel_mask]
        if int(swing_accels.numel()) == 0:
            swing_accels = accel.reshape(-1)
        accel_mean = swing_accels.mean().clamp_min(1.0e-9)
        accel_ratio = _safe_ratio(swing_accels.max(), accel_mean)
    else:
        accel_ratio = 0.0
    boundary_ratio = 0.0
    interval = int(replan_interval_steps)
    if interval > 0 and int(foot.shape[1]) > interval:
        boundary_indices = list(range(interval, int(foot.shape[1]), interval))
        if boundary_indices:
            boundary_steps = torch.stack([step[:, idx - 1] for idx in boundary_indices], dim=1).reshape(-1)
            boundary_ratio = _safe_ratio(boundary_steps.max(), median_step)
    return {
        "fk_swing_foot_step_max_to_median": float(step_ratio),
        "fk_swing_foot_accel_max_to_mean": float(accel_ratio),
        "replan_boundary_fk_foot_step_to_median": float(boundary_ratio),
    }


def reachable_command_frame_endpoint_metrics(
    planned_foot_pos: torch.Tensor,
    fk_foot_pos: torch.Tensor,
    planned_touchdown_pos: torch.Tensor,
    contact_state: torch.Tensor,
    obstacle_xy: torch.Tensor,
    *,
    command: tuple[float, float, float],
    replan_interval_steps: int,
) -> dict[str, float | int]:
    planned = torch.as_tensor(planned_foot_pos, dtype=torch.float32)
    fk = torch.as_tensor(fk_foot_pos, dtype=torch.float32, device=planned.device)
    touchdown = torch.as_tensor(planned_touchdown_pos, dtype=torch.float32, device=planned.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=planned.device)
    obstacle = torch.as_tensor(obstacle_xy, dtype=torch.float32, device=planned.device)
    if touchdown.shape != planned.shape:
        touchdown = planned
    planned_along, _ = _project_command_frame(planned[..., :2], obstacle, command)
    fk_along, _ = _project_command_frame(fk[..., :2], obstacle, command)
    touchdown_along, _ = _project_command_frame(touchdown[..., :2], obstacle, command)
    swing = ~contact
    if int(planned.shape[1]) >= 2:
        planned_step = planned_along[:, 1:] - planned_along[:, :-1]
        fk_step = fk_along[:, 1:] - fk_along[:, :-1]
        swing_step = torch.logical_or(swing[:, 1:], swing[:, :-1])
        planned_backward = torch.relu(-planned_step)[swing_step]
        fk_backward = torch.relu(-fk_step)[swing_step]
        planned_forward = torch.relu(planned_step)[swing_step]
        fk_forward = torch.relu(fk_step)[swing_step]
        if int(planned_backward.numel()) == 0:
            planned_backward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
            fk_backward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
            planned_forward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
            fk_forward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
    else:
        planned_backward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
        fk_backward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
        planned_forward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
        fk_forward = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
    endpoint_backtrack = torch.relu(planned_along - touchdown_along)
    fk_endpoint_backtrack = torch.relu(fk_along - touchdown_along)
    swing_endpoint_backtrack = endpoint_backtrack[swing]
    if int(swing_endpoint_backtrack.numel()) == 0:
        swing_endpoint_backtrack = endpoint_backtrack.reshape(-1)
    planned_fk_along_error = torch.abs(planned_along - fk_along)
    interval = int(replan_interval_steps)
    boundary_backtrack = torch.zeros((1,), dtype=planned.dtype, device=planned.device)
    if interval > 0 and int(planned.shape[1]) > interval:
        boundary_indices = list(range(interval, int(planned.shape[1]), interval))
        if boundary_indices:
            boundary_backtrack = torch.stack(
                [torch.relu(planned_along[:, idx - 1] - planned_along[:, idx]) for idx in boundary_indices],
                dim=1,
            ).reshape(-1)
    worst = torch.argmax(endpoint_backtrack.reshape(-1))
    worst_leg = int((worst % int(endpoint_backtrack.shape[-1])).item()) if int(endpoint_backtrack.numel()) > 0 else -1
    return {
        "planned_swing_along_forward_step_max_m": float(planned_forward.max().item()),
        "planned_swing_along_backward_step_max_m": float(planned_backward.max().item()),
        "fk_swing_along_forward_step_max_m": float(fk_forward.max().item()),
        "fk_swing_along_backward_step_max_m": float(fk_backward.max().item()),
        "touchdown_behind_planned_foot_along_max_m": float(endpoint_backtrack.max().item()),
        "touchdown_behind_fk_foot_along_max_m": float(fk_endpoint_backtrack.max().item()),
        "touchdown_behind_swing_foot_along_max_m": float(swing_endpoint_backtrack.max().item()),
        "planned_vs_fk_along_error_max_m": float(planned_fk_along_error.max().item()),
        "replan_boundary_planned_along_backtrack_max_m": float(boundary_backtrack.max().item()),
        "touchdown_behind_worst_leg": worst_leg,
    }


def reachable_foot_height_relative_to_root_metrics(
    planned_foot_pos: torch.Tensor,
    fk_foot_pos: torch.Tensor,
    root_pos: torch.Tensor,
    contact_state: torch.Tensor,
) -> dict[str, float]:
    planned = torch.as_tensor(planned_foot_pos, dtype=torch.float32)
    fk = torch.as_tensor(fk_foot_pos, dtype=torch.float32, device=planned.device)
    root = torch.as_tensor(root_pos, dtype=torch.float32, device=planned.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=planned.device)
    root_z = root[..., 2].unsqueeze(-1)
    planned_above = planned[..., 2] - root_z
    fk_above = fk[..., 2] - root_z
    swing = ~contact
    planned_swing = planned_above[swing]
    fk_swing = fk_above[swing]
    if int(planned_swing.numel()) == 0:
        planned_swing = planned_above.reshape(-1)
    if int(fk_swing.numel()) == 0:
        fk_swing = fk_above.reshape(-1)
    return {
        "planned_foot_above_root_z_max_m": float(planned_above.max().item()),
        "planned_swing_foot_above_root_z_max_m": float(planned_swing.max().item()),
        "fk_foot_above_root_z_max_m": float(fk_above.max().item()),
        "fk_swing_foot_above_root_z_max_m": float(fk_swing.max().item()),
    }


def reachable_foot_over_arc_metrics(
    fk_foot_pos: torch.Tensor,
    contact_state: torch.Tensor,
    obstacle_xy: torch.Tensor,
    *,
    command: tuple[float, float, float],
    obstacle_height: float,
    clearance: float,
    lane_half_width: float,
) -> dict[str, float | int]:
    foot = torch.as_tensor(fk_foot_pos, dtype=torch.float32)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    obstacle = torch.as_tensor(obstacle_xy, dtype=torch.float32, device=foot.device)
    if foot.ndim != 4 or tuple(foot.shape[-2:]) != (4, 3):
        raise ValueError("fk_foot_pos must have shape [B, T, 4, 3]")
    along, lateral = _project_command_frame(foot[..., :2], obstacle, command)
    z = foot[..., 2]
    over_lane = torch.abs(lateral) <= float(lane_half_width)
    above = z >= float(obstacle_height) + float(clearance)
    front = along < -0.02
    back = along > 0.02
    touchdown_after = torch.logical_and(contact, back)
    clearance_over_top = torch.clamp(z - float(obstacle_height), min=0.0)

    success = False
    lift_then_land = False
    touchdown_ok = False
    min_lateral = float("inf")
    max_clearance = 0.0
    batch, _, legs, _ = foot.shape
    for b in range(batch):
        for leg in range(legs):
            lane_leg = over_lane[b, :, leg]
            if bool(lane_leg.any().item()):
                min_lateral = min(min_lateral, float(torch.abs(lateral[b, :, leg][lane_leg]).min().item()))
                max_clearance = max(max_clearance, float(clearance_over_top[b, :, leg][lane_leg].max().item()))
            crossed = bool(front[b, :, leg].any().item() and back[b, :, leg].any().item())
            lifted = bool(torch.logical_and(lane_leg, above[b, :, leg]).any().item())
            touchdown_frames = torch.nonzero(torch.logical_and(touchdown_after[b, :, leg], lane_leg), as_tuple=False).reshape(-1)
            if crossed and lifted and int(touchdown_frames.numel()) > 0:
                lift_frames = torch.nonzero(torch.logical_and(lane_leg, above[b, :, leg]), as_tuple=False).reshape(-1)
                if int(lift_frames.numel()) > 0 and int(touchdown_frames[-1].item()) > int(lift_frames[0].item()):
                    success = True
                    lift_then_land = True
                    touchdown_ok = True
    if math.isinf(min_lateral):
        min_lateral = float("inf")
    return {
        "fk_foot_over_low_small_success": int(success),
        "fk_foot_over_low_small_min_lateral": float(min_lateral),
        "fk_foot_over_low_small_clearance_max": float(max_clearance),
        "fk_foot_over_low_small_lift_then_land": int(lift_then_land),
        "fk_foot_over_low_small_touchdown_after": int(touchdown_ok),
    }


def reachable_ik_fk_consistency_metrics(
    planned_foot_pos: torch.Tensor,
    fk_foot_pos: torch.Tensor,
    planned_touchdown_pos: torch.Tensor,
    fk_touchdown_pos: torch.Tensor,
    raw_joint_angles: torch.Tensor,
    clamped_joint_angles: torch.Tensor,
) -> dict[str, float]:
    planned_foot = torch.as_tensor(planned_foot_pos, dtype=torch.float32)
    fk_foot = torch.as_tensor(fk_foot_pos, dtype=torch.float32, device=planned_foot.device)
    planned_touchdown = torch.as_tensor(planned_touchdown_pos, dtype=torch.float32, device=planned_foot.device)
    fk_touchdown = torch.as_tensor(fk_touchdown_pos, dtype=torch.float32, device=planned_foot.device)
    raw_joint = torch.as_tensor(raw_joint_angles, dtype=torch.float32, device=planned_foot.device)
    clamped_joint = torch.as_tensor(clamped_joint_angles, dtype=torch.float32, device=planned_foot.device)
    limits = _JOINT_LIMITS.to(device=raw_joint.device, dtype=raw_joint.dtype).view(1, 1, 12, 2)
    lower_violation = torch.relu(limits[..., 0] - raw_joint)
    upper_violation = torch.relu(raw_joint - limits[..., 1])
    calf_delta = torch.abs(clamped_joint[..., 2::3] - raw_joint[..., 2::3])
    return {
        "terminal_planned_vs_fk_foot_error_max": float(torch.linalg.vector_norm(planned_foot - fk_foot, dim=-1).max().item()),
        "touchdown_ik_fk_error_max": float(torch.linalg.vector_norm(planned_touchdown - fk_touchdown, dim=-1).max().item()),
        "raw_ik_joint_limit_violation_max": float(torch.maximum(lower_violation, upper_violation).max().item()),
        "calf_upper_saturation_max": float(calf_delta.max().item()),
    }


def _semantic_small_contact_metrics(fk_foot: torch.Tensor, touchdown: torch.Tensor, contact_state: torch.Tensor, terrain) -> dict[str, float]:
    from extension.batch_mpc_planner.terrain import height_at, semantic_at

    foot = torch.as_tensor(fk_foot, dtype=torch.float32)
    td = torch.as_tensor(touchdown, dtype=torch.float32, device=foot.device)
    contact = torch.as_tensor(contact_state, dtype=torch.bool, device=foot.device)
    foot_sem = torch.as_tensor(semantic_at(terrain, foot[..., :2].reshape(-1, 2)), device=foot.device).reshape(foot.shape[:-1])
    td_sem = torch.as_tensor(semantic_at(terrain, td[..., :2].reshape(-1, 2)), device=foot.device).reshape(td.shape[:-1])
    foot_h = torch.as_tensor(height_at(terrain, foot[..., :2].reshape(-1, 2)), dtype=foot.dtype, device=foot.device).reshape(foot.shape[:-1])
    small_foot = foot_sem == 1
    small_td = td_sem == 1
    penetration = torch.logical_and(small_foot, foot[..., 2] <= foot_h + 1.0e-4)
    stance_small = torch.logical_and(small_foot, contact)
    return {
        "fk_stance_on_small_rate": float(stance_small.to(dtype=torch.float32).mean().item()),
        "fk_touchdown_on_small_rate": float(small_td.to(dtype=torch.float32).mean().item()),
        "fk_foot_small_penetration_rate": float(penetration.to(dtype=torch.float32).mean().item()),
    }


def compute_plane_low_small_fk_metrics(
    *,
    target_foot_pos: torch.Tensor,
    fk_points,
    terrain,
    plane_mask: torch.Tensor | None,
    probe_half_width_m: float,
    probe_count: int,
) -> dict[str, float | int | list[int] | dict[str, int]]:
    from extension.batch_mpc_planner.terrain import height_at, semantic_at

    target = torch.as_tensor(target_foot_pos, dtype=torch.float32)
    device = target.device
    batch, horizon, legs = int(target.shape[0]), int(target.shape[1]), int(target.shape[2])
    plane = torch.ones((batch,), dtype=torch.bool, device=device) if plane_mask is None else torch.as_tensor(plane_mask, dtype=torch.bool, device=device).reshape(-1)
    if int(plane.numel()) == 1 and batch > 1:
        plane = plane.expand(batch)
    count = max(1, int(probe_count))
    offsets = torch.linspace(-float(probe_half_width_m), float(probe_half_width_m), count, dtype=target.dtype, device=device)
    probe = target[..., None, :2].clone().expand(batch, horizon, legs, count, 2).clone()
    probe[..., 1] = probe[..., 1] + offsets.view(1, 1, 1, count)
    probe_sem = semantic_at(terrain, probe.reshape(batch, horizon * legs * count, 2)).reshape(batch, horizon, legs, count).to(device=device)
    crossing_leg_mask = torch.logical_and((probe_sem == 1).any(dim=(1, 3)), plane[:, None])
    crossing_leg_count = int(crossing_leg_mask.sum().item())

    parts = {
        "foot": torch.as_tensor(fk_points.foot_pos_world, dtype=target.dtype, device=device),
        "knee": torch.as_tensor(fk_points.knee_pos_world, dtype=target.dtype, device=device),
        "shank": torch.as_tensor(fk_points.shank_sample_world, dtype=target.dtype, device=device),
    }
    collision_by_part: dict[str, int] = {}
    collision_by_leg_tensor = torch.zeros((batch, legs), dtype=torch.long, device=device)
    total_collision = 0
    min_clearance: torch.Tensor | None = None
    first_frame = horizon
    for name, points in parts.items():
        pts = points.reshape(batch, -1, 3)
        sem = semantic_at(terrain, pts[..., :2]).reshape(batch, -1).to(device=device)
        height = height_at(terrain, pts[..., :2]).reshape(batch, -1).to(dtype=target.dtype, device=device)
        clearance = pts[..., 2] - height
        hit = torch.logical_and(torch.logical_and(sem == 1, clearance < 0.0), plane[:, None])
        semantic_region = torch.logical_and(sem == 1, plane[:, None])
        if name == "shank":
            crossing_flat = crossing_leg_mask[:, None, :, None].expand(batch, horizon, legs, int(points.shape[-2])).reshape(batch, -1)
        else:
            crossing_flat = crossing_leg_mask[:, None, :].expand(batch, horizon, legs).reshape(batch, -1)
        hit = torch.logical_and(hit, crossing_flat)
        semantic_region = torch.logical_and(semantic_region, crossing_flat)
        collision_by_part[name] = int(hit.sum().item())
        total_collision += collision_by_part[name]
        semantic_clearance = torch.where(semantic_region, clearance, torch.full_like(clearance, float("inf")))
        part_min = semantic_clearance.amin()
        min_clearance = part_min if min_clearance is None else torch.minimum(min_clearance, part_min)
        if bool(hit.any().item()):
            if name == "shank":
                hit_leg = hit.reshape(batch, horizon, legs, -1).any(dim=(1, 3))
                hit_frame = hit.reshape(batch, horizon, legs, -1).any(dim=(2, 3))
            else:
                hit_leg = hit.reshape(batch, horizon, legs).any(dim=1)
                hit_frame = hit.reshape(batch, horizon, legs).any(dim=2)
            collision_by_leg_tensor += hit_leg.to(dtype=torch.long)
            frame_ids = torch.nonzero(hit_frame, as_tuple=False)
            if int(frame_ids.numel()) > 0:
                first_frame = min(first_frame, int(frame_ids[:, 1].min().item()))
    target_fk_error = torch.linalg.vector_norm(target - parts["foot"], dim=-1)
    crossing_error = target_fk_error[crossing_leg_mask[:, None, :].expand(batch, horizon, legs)]
    if int(crossing_error.numel()) == 0:
        crossing_error = torch.zeros((1,), dtype=target.dtype, device=device)
    denom = max(1, batch * horizon * legs)
    min_clearance_value = 0.0 if min_clearance is None or not torch.isfinite(min_clearance) else float(min_clearance.item())
    return {
        "plane_env_count": int(plane.sum().item()),
        "crossing_leg_count": crossing_leg_count,
        "crossing_leg_mask": crossing_leg_mask.to(dtype=torch.long).cpu().reshape(-1).tolist(),
        "fk_semantic_collision_count": int(total_collision),
        "fk_semantic_collision_rate": float(total_collision) / float(denom),
        "fk_semantic_collision_by_part": collision_by_part,
        "fk_semantic_collision_by_leg": collision_by_leg_tensor.sum(dim=0).cpu().tolist(),
        "fk_semantic_min_clearance_over_semantic_m": min_clearance_value,
        "fk_semantic_first_collision_frame": int(first_frame if first_frame < horizon else -1),
        "planned_vs_fk_foot_error_crossing_leg_max_m": float(crossing_error.max().item()),
        "planned_vs_fk_foot_error_all_max_m": float(target_fk_error.max().item()),
    }


def compute_segmented_plane_low_small_fk_metrics(
    *,
    target_foot_pos: torch.Tensor,
    fk_points,
    terrains: tuple[object, ...],
    segment_lengths: tuple[int, ...],
    probe_half_width_m: float,
    probe_count: int,
) -> dict[str, float | int | list[int] | dict[str, int]]:
    target = torch.as_tensor(target_foot_pos, dtype=torch.float32)
    if not terrains or not segment_lengths:
        raise ValueError("terrains and segment_lengths must be non-empty")
    if len(terrains) != len(segment_lengths):
        raise ValueError("terrains and segment_lengths must have the same length")

    batch, horizon, legs = int(target.shape[0]), int(target.shape[1]), int(target.shape[2])
    foot = torch.as_tensor(fk_points.foot_pos_world, dtype=target.dtype, device=target.device)
    knee = torch.as_tensor(fk_points.knee_pos_world, dtype=target.dtype, device=target.device)
    shank = torch.as_tensor(fk_points.shank_sample_world, dtype=target.dtype, device=target.device)
    aggregate_collision_by_part = {"foot": 0, "knee": 0, "shank": 0}
    aggregate_collision_by_leg = torch.zeros((legs,), dtype=torch.long)
    crossing_leg_any = torch.zeros((batch, legs), dtype=torch.bool, device=target.device)
    total_collision = 0
    checked_point_count = 0
    min_clearance = float("inf")
    first_collision_frame = -1
    crossing_errors: list[torch.Tensor] = []
    all_error_max = 0.0
    plane_env_count = 0

    start = 0
    for terrain, length in zip(terrains, segment_lengths):
        seg_len = max(0, min(int(length), horizon - start))
        if seg_len <= 0:
            continue
        stop = start + seg_len
        seg_fk_points = SimpleNamespace(
            foot_pos_world=foot[:, start:stop],
            knee_pos_world=knee[:, start:stop],
            shank_sample_world=shank[:, start:stop],
        )
        metrics = compute_plane_low_small_fk_metrics(
            target_foot_pos=target[:, start:stop],
            fk_points=seg_fk_points,
            terrain=terrain,
            plane_mask=getattr(terrain, "is_plane_terrain", None),
            probe_half_width_m=probe_half_width_m,
            probe_count=probe_count,
        )
        plane_env_count = max(plane_env_count, int(metrics["plane_env_count"]))
        crossing_mask = torch.as_tensor(metrics["crossing_leg_mask"], dtype=torch.bool, device=target.device).reshape(batch, legs)
        crossing_leg_any |= crossing_mask
        for name in aggregate_collision_by_part:
            aggregate_collision_by_part[name] += int(metrics["fk_semantic_collision_by_part"][name])
        aggregate_collision_by_leg += torch.as_tensor(metrics["fk_semantic_collision_by_leg"], dtype=torch.long)
        total_collision += int(metrics["fk_semantic_collision_count"])
        checked_point_count += batch * seg_len * legs
        seg_min = float(metrics["fk_semantic_min_clearance_over_semantic_m"])
        if int(metrics["crossing_leg_count"]) > 0 or int(metrics["fk_semantic_collision_count"]) > 0:
            min_clearance = min(min_clearance, seg_min)
        seg_first = int(metrics["fk_semantic_first_collision_frame"])
        if seg_first >= 0 and (first_collision_frame < 0 or start + seg_first < first_collision_frame):
            first_collision_frame = start + seg_first
        seg_error = torch.linalg.vector_norm(target[:, start:stop] - foot[:, start:stop], dim=-1)
        all_error_max = max(all_error_max, float(seg_error.max().item()))
        if bool(crossing_mask.any().item()):
            crossing_errors.append(seg_error[crossing_mask[:, None, :].expand(batch, seg_len, legs)])
        start = stop
        if start >= horizon:
            break

    if crossing_errors:
        crossing_error = torch.cat(crossing_errors)
        crossing_error_max = float(crossing_error.max().item())
    else:
        crossing_error_max = 0.0
    min_clearance_value = 0.0 if min_clearance == float("inf") else float(min_clearance)
    denom = max(1, checked_point_count)
    return {
        "plane_env_count": int(plane_env_count),
        "crossing_leg_count": int(crossing_leg_any.sum().item()),
        "crossing_leg_mask": crossing_leg_any.to(dtype=torch.long).cpu().reshape(-1).tolist(),
        "fk_semantic_collision_count": int(total_collision),
        "fk_semantic_collision_rate": float(total_collision) / float(denom),
        "fk_semantic_collision_by_part": aggregate_collision_by_part,
        "fk_semantic_collision_by_leg": aggregate_collision_by_leg.cpu().tolist(),
        "fk_semantic_min_clearance_over_semantic_m": min_clearance_value,
        "fk_semantic_first_collision_frame": int(first_collision_frame),
        "planned_vs_fk_foot_error_crossing_leg_max_m": crossing_error_max,
        "planned_vs_fk_foot_error_all_max_m": all_error_max,
    }


def _stability_metrics(root_pos: torch.Tensor, root_rpy: torch.Tensor, fk_foot: torch.Tensor) -> dict[str, float]:
    root = torch.as_tensor(root_pos, dtype=torch.float32)
    rpy = torch.as_tensor(root_rpy, dtype=torch.float32, device=root.device)
    foot = torch.as_tensor(fk_foot, dtype=torch.float32, device=root.device)
    rel = foot[..., :2] - root[:, :, None, :2]
    lateral = rel[..., 1]
    spread = torch.amax(foot[..., 1], dim=-1) - torch.amin(foot[..., 1], dim=-1)
    return {
        "root_height_min": float(root[..., 2].min().item()),
        "base_bottom_clearance_min": float((root[..., 2] - 0.10).min().item()),
        "roll_pitch_abs_max": float(torch.abs(rpy[..., :2]).max().item()),
        "foot_lateral_spread_max": float(spread.max().item()),
        "foot_to_root_lateral_offset_max": float(torch.abs(lateral).max().item()),
        "hip_abduction_limit_margin_min": 0.0,
    }


def _result_metrics(result, terrain, obstacle_xy: torch.Tensor, command: tuple[float, float, float], *, obstacle_height: float) -> dict[str, float | int]:
    root = torch.as_tensor(result.root_pos_w, dtype=torch.float32)
    rpy = _root_rpy_from_viewer_result(result).to(dtype=root.dtype, device=root.device)
    planned_foot = torch.as_tensor(result.foot_pos_w, dtype=root.dtype, device=root.device)
    joint = torch.as_tensor(result.joint_angles, dtype=root.dtype, device=root.device)
    fk_foot = fk_feet_from_joint_angles(root, rpy, joint)
    fk_points = fk_leg_points_from_joint_angles(root, rpy, joint, shank_sample_count=2)
    raw_joint = solve_joint_angles_from_trajectory(root, rpy, planned_foot, clamp_to_limits=False)
    clamped_joint = solve_joint_angles_from_trajectory(root, rpy, planned_foot, clamp_to_limits=True)
    planned_touchdown = torch.as_tensor(getattr(result, "planned_touchdown_w", planned_foot), dtype=root.dtype, device=root.device)
    if planned_touchdown.ndim == 4 and planned_touchdown.shape[1] != root.shape[1]:
        planned_touchdown = planned_foot
    fk_touchdown = fk_foot if planned_touchdown.shape == planned_foot.shape else planned_touchdown
    contact = torch.as_tensor(result.contact_state, dtype=torch.bool, device=root.device)
    metrics: dict[str, float | int] = {}
    metrics.update(reachable_command_direction_metrics(root, command=command))
    metrics.update(reachable_swing_continuity_metrics(fk_foot, contact, replan_interval_steps=25))
    metrics.update(
        reachable_command_frame_endpoint_metrics(
            planned_foot,
            fk_foot,
            planned_touchdown,
            contact,
            obstacle_xy,
            command=command,
            replan_interval_steps=25,
        )
    )
    metrics.update(reachable_foot_height_relative_to_root_metrics(planned_foot, fk_foot, root, contact))
    metrics.update(
        reachable_foot_over_arc_metrics(
            fk_foot,
            contact,
            obstacle_xy,
            command=command,
            obstacle_height=float(obstacle_height),
            clearance=0.05,
            lane_half_width=0.10,
        )
    )
    metrics.update(reachable_ik_fk_consistency_metrics(planned_foot, fk_foot, planned_touchdown, fk_touchdown, raw_joint, clamped_joint))
    metrics.update(_semantic_small_contact_metrics(fk_foot, planned_touchdown, contact, terrain))
    segment_terrains = getattr(result, "rolling_segment_terrains", None)
    segment_lengths = getattr(result, "rolling_segment_lengths", None)
    if segment_terrains is not None and segment_lengths is not None:
        metrics.update(
            compute_segmented_plane_low_small_fk_metrics(
                target_foot_pos=planned_foot,
                fk_points=fk_points,
                terrains=tuple(segment_terrains),
                segment_lengths=tuple(int(length) for length in segment_lengths),
                probe_half_width_m=0.06,
                probe_count=3,
            )
        )
    else:
        metrics.update(
            compute_plane_low_small_fk_metrics(
                target_foot_pos=planned_foot,
                fk_points=fk_points,
                terrain=terrain,
                plane_mask=getattr(terrain, "is_plane_terrain", None),
                probe_half_width_m=0.06,
                probe_count=3,
            )
        )
    metrics.update(_stability_metrics(root, rpy, fk_foot))
    initial_foot_error = getattr(result, "rolling_segment_initial_foot_error", None)
    if initial_foot_error is not None:
        err = torch.as_tensor(initial_foot_error, dtype=torch.float32)
        metrics["replan_initial_foot_error_max"] = float(err.max().item()) if int(err.numel()) > 0 else 0.0
    else:
        metrics["replan_initial_foot_error_max"] = -1.0
    initial_touchdown_error = getattr(result, "rolling_segment_initial_touchdown_error", None)
    if initial_touchdown_error is not None:
        td_err = torch.as_tensor(initial_touchdown_error, dtype=torch.float32)
        metrics["replan_initial_touchdown_to_current_foot_error_max"] = (
            float(td_err.max().item()) if int(td_err.numel()) > 0 else 0.0
        )
    else:
        metrics["replan_initial_touchdown_to_current_foot_error_max"] = -1.0
    return metrics


def run_probe(
    *,
    device: str,
    commands: tuple[str, ...],
    variants: tuple[str, ...],
    cycles: int,
    requested_n_frames: int,
    warmup_steps: int,
    longitudinal_offset_m: float,
    lateral_offset_m: float,
    semantic_small_height_m: float | None,
    semantic_small_diameter_m: float | None,
    trace_touchdown_chain: bool,
) -> int:
    runtime = RealViewerRuntimeFixture(
        num_envs=1,
        device=device,
        planner_backend="mpc",
        requested_n_frames=requested_n_frames,
        warmup_steps=warmup_steps,
        task_id="Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0",
        env_cfg_entry_point=(
            "go2_pvcnn.tasks.teacher_elevation_trajectory_mpc_semantic_env_cfg:"
            "TeacherElevationTrajectoryMpcSemanticEnvCfg"
        ),
        semantic_small_height_m=semantic_small_height_m,
        semantic_small_diameter_m=semantic_small_diameter_m,
    )
    rows: list[dict[str, float | int | str]] = []
    try:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(
            json.dumps(
                {
                    "type": "reachable_probe_header",
                    "cuda_visible_devices": cuda_visible_devices,
                    "device": device,
                    "commands": list(commands),
                    "variants": list(variants),
                    "cycles": int(cycles),
                    "requested_n_frames": int(requested_n_frames),
                    "warmup_steps": int(warmup_steps),
                    "semantic_small_height_m": None if semantic_small_height_m is None else float(semantic_small_height_m),
                    "semantic_small_diameter_m": None if semantic_small_diameter_m is None else float(semantic_small_diameter_m),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        anchor = runtime.s4_semantic_course_anchor("small")
        obstacle_xy = torch.tensor(anchor.world_xy, dtype=torch.float32, device=runtime.base_env.device)
        for command_text in commands:
            command_name, command_tuple = _parse_command(command_text)
            for variant in variants:
                runtime.reset()
                start_xy = _command_relative_xy(
                    anchor.world_xy,
                    command_tuple,
                    longitudinal_offset_m=longitudinal_offset_m,
                    lateral_offset_m=lateral_offset_m,
                    device=runtime.base_env.device,
                )
                runtime._write_env0_root_xy(start_xy)
                _set_env0_yaw(runtime, _command_heading_yaw(command_tuple))
                runtime._sync_targeted_scan_pose()
                state = runtime._single_env_state()
                for cycle in range(int(cycles)):
                    terrain = runtime._single_env_terrain()
                    planning_command = torch.tensor([command_tuple], dtype=torch.float64, device=runtime.base_env.device)
                    candidate_cfg = reachable_cfg_for_variant(runtime.mpc_planner_cfg, variant, command=command_tuple)
                    probe_seed = _semantic_probe_seed(
                        semantic_class="small",
                        command_name=command_name,
                        cycle=cycle,
                        effective_candidate=variant,
                    )
                    torch.manual_seed(probe_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(probe_seed)
                    if trace_touchdown_chain:
                        trace_row = reachable_touchdown_chain_trace(
                            terrain,
                            state,
                            planning_command,
                            candidate_cfg,
                            obstacle_xy,
                            command_tuple,
                        )
                        trace_row.update(
                            {
                                "command": command_name,
                                "variant": variant,
                                "cycle": int(cycle),
                                "semantic_probe_seed": int(probe_seed),
                            }
                        )
                        print(json.dumps(trace_row, sort_keys=True), flush=True)
                    with _patched_reachable_loss_for_variant(
                        variant,
                        command=planning_command,
                        obstacle_xy=obstacle_xy.unsqueeze(0),
                        obstacle_height=float(anchor.target_height),
                    ):
                        result = _plan_rolling_viewer_trajectory(
                            runtime,
                            terrain=terrain,
                            state=state,
                            command=planning_command,
                            total_frames=runtime.requested_n_frames,
                            candidate_cfg=candidate_cfg,
                            effective_candidate_variant=variant,
                            trace_terminal=False,
                        )
                    row = {
                        "type": "reachable_crossing_cycle",
                        "cuda_visible_devices": cuda_visible_devices,
                        "device": device,
                        "command": command_name,
                        "command_vx": float(command_tuple[0]),
                        "command_vy": float(command_tuple[1]),
                        "command_wz": float(command_tuple[2]),
                        "variant": variant,
                        "cycle": int(cycle),
                        "replan_count": int(cycle + 1),
                        "requested_n_frames": int(runtime.requested_n_frames),
                        "horizon": int(getattr(result, "root_pos_w").shape[1]),
                        "semantic_anchor_x": float(anchor.world_xy[0]),
                        "semantic_anchor_y": float(anchor.world_xy[1]),
                        "semantic_target_diameter": float(anchor.target_diameter),
                        "semantic_target_height": float(anchor.target_height),
                        "semantic_probe_seed": int(probe_seed),
                    }
                    row.update(_result_metrics(result, terrain, obstacle_xy, command_tuple, obstacle_height=float(anchor.target_height)))
                    row["terrain_is_plane"] = int(row.get("plane_env_count", 0) > 0)
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                    refresh_targeted_scanner_pose(runtime.base_env, runtime.scanner, minimum_steps=1, extra_steps=2)
                    state = runtime._single_env_state()
        if rows:
            print(
                json.dumps(
                    {
                        "type": "reachable_probe_summary",
                        "cuda_visible_devices": cuda_visible_devices,
                        "device": device,
                        "cycle_count": len(rows),
                        "max_terminal_planned_vs_fk_foot_error": max(
                            float(row["terminal_planned_vs_fk_foot_error_max"]) for row in rows
                        ),
                        "max_touchdown_ik_fk_error": max(float(row["touchdown_ik_fk_error_max"]) for row in rows),
                        "max_replan_initial_foot_error": max(float(row["replan_initial_foot_error_max"]) for row in rows),
                        "max_replan_initial_touchdown_to_current_foot_error": max(
                            float(row["replan_initial_touchdown_to_current_foot_error_max"]) for row in rows
                        ),
                        "max_fk_stance_on_small_rate": max(float(row["fk_stance_on_small_rate"]) for row in rows),
                        "max_fk_touchdown_on_small_rate": max(float(row["fk_touchdown_on_small_rate"]) for row in rows),
                        "max_fk_foot_small_penetration_rate": max(float(row["fk_foot_small_penetration_rate"]) for row in rows),
                        "fk_foot_over_low_small_success_count": sum(
                            int(row["fk_foot_over_low_small_success"]) for row in rows
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        runtime.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--commands", default=",".join(DEFAULT_COMMANDS))
    parser.add_argument("--variants", default="baseline")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--requested-n-frames", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--longitudinal-offset-m", type=float, default=-0.35)
    parser.add_argument("--lateral-offset-m", type=float, default=0.0)
    parser.add_argument("--semantic-small-height-m", type=float, default=None)
    parser.add_argument("--semantic-small-diameter-m", type=float, default=None)
    parser.add_argument("--trace-touchdown-chain", action="store_true")
    args = parser.parse_args()
    commands = tuple(item.strip() for item in str(args.commands).split(",") if item.strip())
    variants = tuple(item.strip() for item in str(args.variants).split(",") if item.strip())
    return run_probe(
        device=str(args.device),
        commands=commands,
        variants=variants,
        cycles=int(args.cycles),
        requested_n_frames=int(args.requested_n_frames),
        warmup_steps=int(args.warmup_steps),
        longitudinal_offset_m=float(args.longitudinal_offset_m),
        lateral_offset_m=float(args.lateral_offset_m),
        semantic_small_height_m=args.semantic_small_height_m,
        semantic_small_diameter_m=args.semantic_small_diameter_m,
        trace_touchdown_chain=bool(args.trace_touchdown_chain),
    )


if __name__ == "__main__":
    raise SystemExit(main())
