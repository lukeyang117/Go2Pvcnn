"""Parametric trajectory helpers for the batch MPC backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def bounded_unit_interval(raw: Tensor, *, low: float, high: float) -> Tensor:
    return float(low) + (float(high) - float(low)) * torch.sigmoid(raw)


def command_frame_axes(command: Tensor, root_yaw: Tensor, *, linear_eps: float) -> tuple[Tensor, Tensor, Tensor]:
    cmd = torch.as_tensor(command)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    xy = cmd[:, :2]
    speed = torch.linalg.vector_norm(xy, dim=-1)
    active = speed > float(linear_eps)
    cmd_forward = xy / speed.clamp_min(1.0e-6).unsqueeze(-1)
    yaw = torch.as_tensor(root_yaw, dtype=cmd.dtype, device=cmd.device).reshape(-1)
    yaw_forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    forward = torch.where(active.unsqueeze(-1), cmd_forward, yaw_forward)
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    return forward, left, active


def cubic_bezier(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, phase: Tensor) -> Tensor:
    t = torch.as_tensor(phase, dtype=p0.dtype, device=p0.device)
    view_shape = (1, int(t.numel())) + (1,) * (p0.ndim - 1)
    t = t.reshape(view_shape)
    one = 1.0 - t
    return (
        one.pow(3) * p0.unsqueeze(1)
        + 3.0 * one.pow(2) * t * p1.unsqueeze(1)
        + 3.0 * one * t.pow(2) * p2.unsqueeze(1)
        + t.pow(3) * p3.unsqueeze(1)
    )


@dataclass
class MpcParametricVariables:
    touchdown_delta_raw: Tensor
    swing_clearance_raw: Tensor
    bezier_ab_raw: Tensor
    lateral_bias_raw: Tensor
    root_goal_delta_raw: Tensor
    root_bezier_raw: Tensor
    root_lateral_bias_raw: Tensor
    root_height_offset_raw: Tensor
    swing_center_raw: Tensor
    swing_width_raw: Tensor
    diagonal_phase_raw: Tensor

    def parameters(self) -> list[Tensor]:
        return [
            self.touchdown_delta_raw,
            self.swing_clearance_raw,
            self.bezier_ab_raw,
            self.lateral_bias_raw,
            self.root_goal_delta_raw,
            self.root_bezier_raw,
            self.root_lateral_bias_raw,
            self.root_height_offset_raw,
            self.swing_center_raw,
            self.swing_width_raw,
            self.diagonal_phase_raw,
        ]


def _optim_zeros(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device).requires_grad_(True)


def init_parametric_variables(state, command: Tensor, *, horizon: int) -> MpcParametricVariables:
    del command, horizon
    root = torch.as_tensor(state.root_pos)
    batch = int(root.shape[0])
    dtype = root.dtype
    device = root.device
    return MpcParametricVariables(
        touchdown_delta_raw=_optim_zeros((batch, 4, 2), dtype=dtype, device=device),
        swing_clearance_raw=_optim_zeros((batch, 4), dtype=dtype, device=device),
        bezier_ab_raw=_optim_zeros((batch, 4, 2), dtype=dtype, device=device),
        lateral_bias_raw=_optim_zeros((batch, 4, 2), dtype=dtype, device=device),
        root_goal_delta_raw=_optim_zeros((batch, 2), dtype=dtype, device=device),
        root_bezier_raw=_optim_zeros((batch, 2), dtype=dtype, device=device),
        root_lateral_bias_raw=_optim_zeros((batch, 2), dtype=dtype, device=device),
        root_height_offset_raw=_optim_zeros((batch,), dtype=dtype, device=device),
        swing_center_raw=_optim_zeros((batch, 4), dtype=dtype, device=device),
        swing_width_raw=_optim_zeros((batch, 4), dtype=dtype, device=device),
        diagonal_phase_raw=_optim_zeros((batch,), dtype=dtype, device=device),
    )


@dataclass(frozen=True)
class DecodedParametricTrajectory:
    root_pos: Tensor
    root_rpy: Tensor
    target_foot_pos: Tensor
    touchdown_w: Tensor
    swing_center: Tensor
    swing_width: Tensor
    contact_prob: Tensor
    swing_prob: Tensor


def _phase(horizon: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.linspace(0.0, 1.0, int(horizon), dtype=dtype, device=device)


def _padded_command(command: Tensor, *, batch: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    cmd = torch.as_tensor(command, dtype=dtype, device=device)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((batch, 3 - int(cmd.shape[-1])), dtype=dtype, device=device)
        cmd = torch.cat((cmd, pad), dim=-1)
    return cmd[:, :3]


def decode_parametric_trajectory(
    state,
    terrain,
    command: Tensor,
    variables: MpcParametricVariables,
    *,
    horizon: int,
) -> DecodedParametricTrajectory:
    from .terrain import height_at

    root0 = torch.as_tensor(state.root_pos)
    rpy0 = torch.as_tensor(state.root_rpy, dtype=root0.dtype, device=root0.device)
    foot0 = torch.as_tensor(state.foot_pos, dtype=root0.dtype, device=root0.device)
    batch = int(root0.shape[0])
    dtype = root0.dtype
    device = root0.device
    h = int(horizon)
    phase = _phase(h, dtype=dtype, device=device)
    cmd = _padded_command(command, batch=batch, dtype=dtype, device=device)
    forward, left, _linear_active = command_frame_axes(cmd, rpy0[:, 2], linear_eps=1.0e-4)

    touchdown_delta = 0.35 * torch.tanh(variables.touchdown_delta_raw)
    touchdown_xy = foot0[..., :2] + forward[:, None, :] * touchdown_delta[..., 0:1] + left[:, None, :] * touchdown_delta[..., 1:2]
    touchdown_z = height_at(terrain, touchdown_xy).to(dtype=dtype, device=device)
    touchdown_w = torch.cat((touchdown_xy, touchdown_z.unsqueeze(-1)), dim=-1)

    ab = bounded_unit_interval(variables.bezier_ab_raw, low=0.15, high=0.85)
    a = ab[..., 0:1]
    b = ab[..., 1:2]
    lateral_bias = 0.20 * torch.tanh(variables.lateral_bias_raw)
    step = touchdown_xy - foot0[..., :2]
    length = torch.linalg.vector_norm(step, dim=-1, keepdim=True).clamp_min(1.0e-6)
    p0 = foot0[..., :2]
    p3 = touchdown_xy
    p1 = p0 + forward[:, None, :] * (a * length) + left[:, None, :] * lateral_bias[..., 0:1]
    p2 = p3 - forward[:, None, :] * (b * length) + left[:, None, :] * lateral_bias[..., 1:2]
    foot_xy = cubic_bezier(p0, p1, p2, p3, phase)
    terrain_z = height_at(terrain, foot_xy.reshape(batch, h * 4, 2)).reshape(batch, h, 4).to(dtype=dtype, device=device)
    clearance = 0.04 + 0.16 * torch.sigmoid(variables.swing_clearance_raw)
    base_z = torch.lerp(foot0[:, None, :, 2], touchdown_z[:, None, :], phase.view(1, h, 1))
    arc = 4.0 * phase.view(1, h, 1) * (1.0 - phase.view(1, h, 1)) * clearance[:, None, :]
    foot_z = torch.maximum(base_z + arc, terrain_z + 0.025)
    foot_z = foot_z.clone()
    foot_z[:, 0, :] = foot0[..., 2]
    target_foot_pos = torch.cat((foot_xy, foot_z.unsqueeze(-1)), dim=-1)

    root_goal_delta = torch.stack(
        (
            0.30 + 0.50 * torch.sigmoid(variables.root_goal_delta_raw[:, 0]),
            0.25 * torch.tanh(variables.root_goal_delta_raw[:, 1]),
        ),
        dim=-1,
    )
    root_goal_xy = root0[:, :2] + forward * root_goal_delta[:, 0:1] + left * root_goal_delta[:, 1:2]
    root_c = bounded_unit_interval(variables.root_bezier_raw, low=0.15, high=0.85)
    root_step = root_goal_xy - root0[:, :2]
    root_len = torch.linalg.vector_norm(root_step, dim=-1, keepdim=True).clamp_min(1.0e-6)
    root_lat = 0.20 * torch.tanh(variables.root_lateral_bias_raw)
    r0 = root0[:, :2]
    r3 = root_goal_xy
    r1 = r0 + forward * (root_c[:, 0:1] * root_len) + left * root_lat[:, 0:1]
    r2 = r3 - forward * (root_c[:, 1:2] * root_len) + left * root_lat[:, 1:2]
    root_xy = cubic_bezier(r0, r1, r2, r3, phase)
    root_ground = height_at(terrain, root_xy).to(dtype=dtype, device=device)
    root_z = root_ground + 0.32 + 0.06 * torch.tanh(variables.root_height_offset_raw).view(batch, 1)
    root_z = root_z.clone()
    root_z[:, 0] = root0[:, 2]
    root_pos = torch.cat((root_xy, root_z.unsqueeze(-1)), dim=-1)
    root_rpy = rpy0[:, None, :].expand(batch, h, 3).clone()
    root_rpy[..., 2] = torch.lerp(rpy0[:, None, 2], rpy0[:, None, 2] + cmd[:, None, 2] * 0.5, phase.view(1, h))

    base_center = torch.tensor((0.75, 0.25, 0.25, 0.75), dtype=dtype, device=device).view(1, 4)
    swing_center = torch.remainder(base_center + 0.20 * torch.tanh(variables.swing_center_raw), 1.0)
    swing_width = bounded_unit_interval(variables.swing_width_raw, low=0.30, high=0.70)
    frame_phase = phase.view(1, h, 1)
    dist = torch.abs(torch.remainder(frame_phase - swing_center[:, None, :] + 0.5, 1.0) - 0.5)
    swing_prob = torch.sigmoid(40.0 * (0.5 * swing_width[:, None, :] - dist))
    contact_prob = 1.0 - swing_prob
    return DecodedParametricTrajectory(
        root_pos=root_pos,
        root_rpy=root_rpy,
        target_foot_pos=target_foot_pos,
        touchdown_w=touchdown_w,
        swing_center=swing_center,
        swing_width=swing_width,
        contact_prob=contact_prob,
        swing_prob=swing_prob,
    )


__all__ = [
    "DecodedParametricTrajectory",
    "MpcParametricVariables",
    "bounded_unit_interval",
    "command_frame_axes",
    "cubic_bezier",
    "decode_parametric_trajectory",
    "init_parametric_variables",
]
