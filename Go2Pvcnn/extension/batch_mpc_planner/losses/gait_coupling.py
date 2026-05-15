"""Gait-coupling losses for continuous swing-window MPC."""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import MpcRuntimeCfg
from ..terrain import height_at, semantic_at, slope_at
from ..types import MpcPlannerTerrain, MpcRobotState
from .terrain_clearance import finite_horizon_touchdown_phase, sample_time


def _circular_signed_delta(a: Tensor, b: Tensor) -> Tensor:
    return torch.remainder(a - b + 0.5, 1.0) - 0.5


def _circular_abs_delta(a: Tensor, b: Tensor) -> Tensor:
    return torch.abs(_circular_signed_delta(a, b))


def _forward_phase_distance(start: Tensor, end: Tensor) -> Tensor:
    return torch.remainder(end - start, 1.0)


def _circular_mean2(a: Tensor, b: Tensor) -> Tensor:
    angles = torch.stack((a, b), dim=0) * (2.0 * torch.pi)
    vec = torch.stack((torch.cos(angles).mean(dim=0), torch.sin(angles).mean(dim=0)), dim=-1)
    return torch.remainder(torch.atan2(vec[..., 1], vec[..., 0]) / (2.0 * torch.pi), 1.0)


def _rotate_world_xy_to_body(xy: Tensor, yaw: Tensor) -> Tensor:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    return torch.stack((cy * xy[..., 0] + sy * xy[..., 1], -sy * xy[..., 0] + cy * xy[..., 1]), dim=-1)


def swing_window_loss(swing_width: Tensor, nominal: dict[str, Tensor], runtime_cfg: MpcRuntimeCfg) -> Tensor:
    width_min = float(runtime_cfg.swing_window_min_width)
    width_max = float(runtime_cfg.swing_window_max_width)
    bounds = torch.relu(width_min - swing_width).square() + torch.relu(swing_width - width_max).square()
    nominal_width = nominal["swing_width"].to(dtype=swing_width.dtype, device=swing_width.device)
    prior = torch.square(swing_width - nominal_width)
    return (bounds + 0.2 * prior).mean(dim=-1)


def diagonal_pair_loss(swing_center: Tensor, swing_width: Tensor) -> Tensor:
    fl, fr, rl, rr = 0, 1, 2, 3
    pair_a = _circular_abs_delta(swing_center[:, fl], swing_center[:, rr])
    pair_b = _circular_abs_delta(swing_center[:, fr], swing_center[:, rl])
    center_a = _circular_mean2(swing_center[:, fl], swing_center[:, rr])
    center_b = _circular_mean2(swing_center[:, fr], swing_center[:, rl])
    half_sep = torch.square(_circular_abs_delta(center_a, center_b) - 0.5)
    width_match = (
        torch.square(swing_width[:, fl] - swing_width[:, rr])
        + torch.square(swing_width[:, fr] - swing_width[:, rl])
    )
    return pair_a.square() + pair_b.square() + half_sep + 0.25 * width_match


def phase_prior_loss(swing_center: Tensor, swing_width: Tensor, nominal: dict[str, Tensor]) -> Tensor:
    center_prior = nominal["swing_center"].to(dtype=swing_center.dtype, device=swing_center.device)
    width_prior = nominal["swing_width"].to(dtype=swing_width.dtype, device=swing_width.device)
    center = _circular_abs_delta(swing_center, center_prior).square().mean(dim=-1)
    width = torch.square(swing_width - width_prior).mean(dim=-1)
    return center + width


def root_foot_center_loss(root_pos: Tensor, foot_pos: Tensor) -> Tensor:
    foot_center_xy = foot_pos[..., :2].mean(dim=2)
    return torch.linalg.vector_norm(root_pos[..., :2] - foot_center_xy, dim=-1).mean(dim=1)


def root_height_loss(root_pos: Tensor, nominal: dict[str, Tensor]) -> Tensor:
    nominal_root = nominal["root_pos"].to(dtype=root_pos.dtype, device=root_pos.device)
    return torch.abs(root_pos[..., 2] - nominal_root[..., 2]).mean(dim=1)


def support_plane_roll_pitch_loss(root_rpy: Tensor, foot_pos: Tensor, contact_prob: Tensor, *, swing_weight: float) -> Tensor:
    weights = float(swing_weight) + (1.0 - float(swing_weight)) * contact_prob.to(dtype=foot_pos.dtype)
    yaw = root_rpy[..., 2].unsqueeze(-1)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    x_w = foot_pos[..., 0]
    y_w = foot_pos[..., 1]
    x = cy * x_w + sy * y_w
    y = -sy * x_w + cy * y_w
    z = foot_pos[..., 2]
    ones = torch.ones_like(x)
    a = torch.stack((x, y, ones), dim=-1)
    w = weights.unsqueeze(-1)
    ata = torch.einsum("btli,btlj->btij", a * w, a)
    atz = torch.einsum("btli,btl->bti", a * w, z)
    eye = torch.eye(3, dtype=foot_pos.dtype, device=foot_pos.device).view(1, 1, 3, 3)
    coeff = torch.linalg.solve(ata + 1.0e-4 * eye, atz.unsqueeze(-1)).squeeze(-1)
    slope_x = coeff[..., 0]
    slope_y = coeff[..., 1]
    est_roll = torch.atan(slope_y)
    est_pitch = -torch.atan(slope_x)
    est = torch.stack((est_roll, est_pitch), dim=-1)
    return torch.linalg.vector_norm(root_rpy[..., :2] - est, dim=-1).mean(dim=1)


def swing_center_urgency_order_loss(
    swing_center: Tensor,
    swing_width: Tensor,
    state: MpcRobotState,
    command: Tensor,
    runtime_cfg: MpcRuntimeCfg,
    *,
    terrain: MpcPlannerTerrain | None = None,
    nominal: dict[str, Tensor] | None = None,
) -> Tensor:
    root = torch.as_tensor(state.root_pos, dtype=swing_center.dtype, device=swing_center.device)
    rpy = torch.as_tensor(state.root_rpy, dtype=swing_center.dtype, device=swing_center.device)
    foot = torch.as_tensor(state.foot_pos, dtype=swing_center.dtype, device=swing_center.device)
    cmd = torch.as_tensor(command, dtype=swing_center.dtype, device=swing_center.device)[:, :3]
    rel = foot[..., :2] - root[:, None, :2]
    foot_body = _rotate_world_xy_to_body(rel, rpy[:, None, 2])
    horizon_time = float(runtime_cfg.dt) * float(runtime_cfg.horizon_steps)
    step_bias = float(runtime_cfg.nominal_stride_scale) * horizon_time * cmd[:, None, :2]
    yaw_bias = (
        float(runtime_cfg.nominal_yaw_stride_scale)
        * horizon_time
        * cmd[:, None, 2:3]
        * torch.stack((-foot_body[..., 1], foot_body[..., 0]), dim=-1)
    )
    expected = step_bias + yaw_bias
    disp = torch.linalg.vector_norm(expected, dim=-1)
    reach = torch.relu(torch.linalg.vector_norm(foot_body + expected, dim=-1) - 0.42)
    terrain_proxy = torch.zeros_like(disp)
    if terrain is not None and nominal is not None and "touchdown_target_w" in nominal:
        touchdown_w = nominal["touchdown_target_w"].to(dtype=swing_center.dtype, device=swing_center.device)
        touchdown_xy = touchdown_w[..., :2]
        touchdown_z = touchdown_w[..., 2]
        terrain_z = height_at(terrain, touchdown_xy).to(dtype=swing_center.dtype, device=swing_center.device)
        touchdown_slope = slope_at(terrain, touchdown_xy).to(dtype=swing_center.dtype, device=swing_center.device)
        semantic = semantic_at(terrain, touchdown_xy)
        semantic_pen = (semantic == 1).to(dtype=swing_center.dtype, device=swing_center.device) + 3.0 * (
            semantic >= 2
        ).to(dtype=swing_center.dtype, device=swing_center.device)
        height_pen = torch.relu(torch.abs(touchdown_z - terrain_z) - 0.03)
        terrain_proxy = semantic_pen + height_pen + torch.relu(touchdown_slope - 0.6)
    urgency = (
        disp
        + float(runtime_cfg.swing_center_reachability_weight) * reach
        + float(runtime_cfg.swing_center_touchdown_proxy_weight) * terrain_proxy
    )
    urgency_a = urgency[:, 0] + urgency[:, 3]
    urgency_b = urgency[:, 1] + urgency[:, 2]
    pair_weight = torch.softmax(
        torch.stack((urgency_a, urgency_b), dim=-1) / float(runtime_cfg.swing_center_urgency_temperature),
        dim=-1,
    )
    swing_start = torch.remainder(swing_center - 0.5 * swing_width, 1.0)
    start_a = _circular_mean2(swing_start[:, 0], swing_start[:, 3])
    start_b = _circular_mean2(swing_start[:, 1], swing_start[:, 2])
    early_a = _forward_phase_distance(torch.zeros_like(start_a), start_a)
    early_b = _forward_phase_distance(torch.zeros_like(start_b), start_b)
    return pair_weight[:, 0] * early_a + pair_weight[:, 1] * early_b


def swing_direction_loss(
    root_pos: Tensor,
    root_rpy: Tensor,
    foot_pos: Tensor,
    swing_center: Tensor,
    swing_width: Tensor,
    command: Tensor,
    runtime_cfg: MpcRuntimeCfg,
) -> Tensor:
    swing_start = torch.remainder(swing_center - 0.5 * swing_width, 1.0)
    swing_end = finite_horizon_touchdown_phase(swing_center, swing_width)
    foot_start = sample_time(foot_pos, swing_start)
    foot_end = sample_time(foot_pos, swing_end, cyclic=False)
    root_start = sample_time(root_pos.unsqueeze(2).expand(-1, -1, 4, -1), swing_start)
    root_end = sample_time(root_pos.unsqueeze(2).expand(-1, -1, 4, -1), swing_end, cyclic=False)
    yaw_start = sample_time(root_rpy[..., 2:3].unsqueeze(2).expand(-1, -1, 4, -1), swing_start).squeeze(-1)
    yaw_end = sample_time(root_rpy[..., 2:3].unsqueeze(2).expand(-1, -1, 4, -1), swing_end, cyclic=False).squeeze(-1)
    start_body = _rotate_world_xy_to_body(foot_start[..., :2] - root_start[..., :2], yaw_start)
    end_body = _rotate_world_xy_to_body(foot_end[..., :2] - root_end[..., :2], yaw_end)
    cmd = torch.as_tensor(command, dtype=foot_pos.dtype, device=foot_pos.device)[:, :3]
    horizon_time = float(runtime_cfg.dt) * float(runtime_cfg.horizon_steps)
    expected = float(runtime_cfg.nominal_stride_scale) * horizon_time * cmd[:, None, :2]
    expected = expected + (
        float(runtime_cfg.nominal_yaw_stride_scale)
        * horizon_time
        * cmd[:, None, 2:3]
        * torch.stack((-start_body[..., 1], start_body[..., 0]), dim=-1)
    )
    err = (end_body - start_body) - expected
    return torch.linalg.vector_norm(err, dim=-1).mean(dim=-1)


__all__ = [
    "diagonal_pair_loss",
    "phase_prior_loss",
    "root_foot_center_loss",
    "root_height_loss",
    "support_plane_roll_pitch_loss",
    "swing_center_urgency_order_loss",
    "swing_direction_loss",
    "swing_window_loss",
]
