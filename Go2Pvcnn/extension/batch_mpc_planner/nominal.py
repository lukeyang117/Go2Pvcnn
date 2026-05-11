"""Nominal trajectory builders for the MPC optimizer."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import MpcRuntimeCfg
from .types import MpcRobotState


def _as_command(command: Tensor, *, device: torch.device) -> Tensor:
    cmd = torch.as_tensor(command, dtype=torch.float32, device=device)
    if cmd.ndim != 2:
        raise ValueError(f"command must have shape [B, C], got {tuple(cmd.shape)}")
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((cmd.shape[0], 3 - cmd.shape[-1]), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    return cmd[:, :3]


def build_nominal_trajectory(
    state: MpcRobotState,
    command: Tensor,
    runtime_cfg: MpcRuntimeCfg,
) -> dict[str, Tensor]:
    """Build a differentiable nominal seed for root/foot/contact."""
    root_pos0 = torch.as_tensor(state.root_pos, dtype=torch.float32)
    root_rpy0 = torch.as_tensor(state.root_rpy, dtype=torch.float32, device=root_pos0.device)
    foot_pos0 = torch.as_tensor(state.foot_pos, dtype=torch.float32, device=root_pos0.device)
    cmd = _as_command(command, device=root_pos0.device)
    batch = int(root_pos0.shape[0])
    horizon = int(runtime_cfg.horizon_steps)
    dt = float(runtime_cfg.dt)

    t = torch.arange(horizon, dtype=root_pos0.dtype, device=root_pos0.device).view(1, horizon, 1)
    root_pos = root_pos0.unsqueeze(1).expand(batch, horizon, 3).clone()
    root_rpy = root_rpy0.unsqueeze(1).expand(batch, horizon, 3).clone()
    root_pos[..., 0] = root_pos[..., 0] + t[..., 0] * dt * cmd[:, 0:1]
    root_pos[..., 1] = root_pos[..., 1] + t[..., 0] * dt * cmd[:, 1:2]
    root_rpy[..., 2] = root_rpy[..., 2] + t[..., 0] * dt * cmd[:, 2:3]

    offsets = torch.as_tensor(runtime_cfg.leg_phase_offsets, dtype=root_pos.dtype, device=root_pos.device).view(1, 1, 4)
    gait_phase = (t * float(runtime_cfg.step_freq) * dt + offsets).expand(batch, horizon, 4) % 1.0
    contact_bool = gait_phase < float(runtime_cfg.duty_factor)
    contact_logits = torch.where(
        contact_bool,
        torch.full_like(gait_phase, 2.0),
        torch.full_like(gait_phase, -2.0),
    )
    foot_rel0_world = foot_pos0 - root_pos0.unsqueeze(1)  # [B,4,3]
    yaw0 = root_rpy0[:, 2]  # [B]
    cy0 = torch.cos(yaw0).unsqueeze(-1)
    sy0 = torch.sin(yaw0).unsqueeze(-1)
    rel0_xy = foot_rel0_world[..., :2]
    # Rotate world relative foot offsets into the initial body frame.
    foot_rel0_body_xy = torch.stack(
        (
            cy0 * rel0_xy[..., 0] + sy0 * rel0_xy[..., 1],
            -sy0 * rel0_xy[..., 0] + cy0 * rel0_xy[..., 1],
        ),
        dim=-1,
    )
    foot_rel_body = torch.zeros((batch, horizon, 4, 3), dtype=root_pos.dtype, device=root_pos.device)
    foot_rel_body[..., :2] = foot_rel0_body_xy.unsqueeze(1).expand(batch, horizon, 4, 2)
    foot_rel_body[..., 2] = foot_rel0_world[..., 2].unsqueeze(1).expand(batch, horizon, 4)
    stride_horizon_s = max(1.0e-6, 1.0 / float(runtime_cfg.step_freq))
    cmd_xy = cmd[:, :2]
    lin_speed = torch.linalg.vector_norm(cmd_xy, dim=-1, keepdim=True)
    abs_yaw = torch.abs(cmd[:, 2:3])
    eps = torch.full_like(abs_yaw, 1.0e-6)
    yaw_frac = abs_yaw / (abs_yaw + lin_speed + eps)
    backward_scale = torch.where(
        cmd[:, 0:1] < 0.0,
        torch.full_like(cmd[:, 0:1], float(runtime_cfg.nominal_backward_stride_scale)),
        torch.ones_like(cmd[:, 0:1]),
    )
    yaw_atten = 1.0 - float(runtime_cfg.nominal_yaw_stride_atten) * yaw_frac
    command_stride_scale = backward_scale * yaw_atten

    raw_stride_xy = cmd[:, None, :2] * (stride_horizon_s * float(runtime_cfg.nominal_stride_scale) * command_stride_scale.unsqueeze(-1))
    stride_norm = torch.linalg.vector_norm(raw_stride_xy, dim=-1, keepdim=True)
    max_stride = float(runtime_cfg.nominal_max_stride_m)
    stride_scale = torch.clamp(max_stride / torch.clamp(stride_norm, min=1.0e-6), max=1.0)
    stride_xy = raw_stride_xy * stride_scale
    yaw_stride = cmd[:, 2:3] * (stride_horizon_s * float(runtime_cfg.nominal_yaw_stride_scale) * yaw_atten)
    rel0_perp_xy = torch.stack(
        (
            -foot_rel0_body_xy[..., 1],
            foot_rel0_body_xy[..., 0],
        ),
        dim=-1,
    )
    yaw_stride_xy = yaw_stride[:, None, :] * rel0_perp_xy
    swing_start = float(runtime_cfg.duty_factor)
    swing_span = max(1.0e-6, 1.0 - swing_start)
    swing_phase = torch.clamp((gait_phase - swing_start) / swing_span, min=0.0, max=1.0)
    smooth_phase = swing_phase * swing_phase * (3.0 - 2.0 * swing_phase)
    centered_phase = smooth_phase - 0.5
    nominal_step_xy = stride_xy.unsqueeze(2) + yaw_stride_xy.unsqueeze(1)
    foot_rel_body[..., :2] = foot_rel_body[..., :2] + centered_phase.unsqueeze(-1) * nominal_step_xy
    foot_rel_body[..., 2] = foot_rel_body[..., 2] + torch.sin(torch.pi * swing_phase) * float(runtime_cfg.nominal_swing_height_m)

    yaw_t = root_rpy[..., 2]  # [B,T]
    cy = torch.cos(yaw_t).unsqueeze(-1)
    sy = torch.sin(yaw_t).unsqueeze(-1)
    rel_body_xy = foot_rel_body[..., :2]
    foot_rel_world_xy = torch.stack(
        (
            cy * rel_body_xy[..., 0] - sy * rel_body_xy[..., 1],
            sy * rel_body_xy[..., 0] + cy * rel_body_xy[..., 1],
        ),
        dim=-1,
    )
    foot_rel_world = torch.cat((foot_rel_world_xy, foot_rel_body[..., 2:3]), dim=-1)
    foot_pos = root_pos.unsqueeze(2) + foot_rel_world

    return {
        "root_pos": root_pos,
        "root_rpy": root_rpy,
        "foot_pos": foot_pos,
        "contact_logits": contact_logits,
    }


__all__ = ["build_nominal_trajectory"]
