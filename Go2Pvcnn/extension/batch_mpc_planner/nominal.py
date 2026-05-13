"""Nominal trajectory builders for the MPC optimizer."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import MpcRuntimeCfg
from .types import MpcFootholdMemory, MpcRobotState


def _as_command(command: Tensor, *, device: torch.device) -> Tensor:
    cmd = torch.as_tensor(command, dtype=torch.float32, device=device)
    if cmd.ndim != 2:
        raise ValueError(f"command must have shape [B, C], got {tuple(cmd.shape)}")
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((cmd.shape[0], 3 - cmd.shape[-1]), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    return cmd[:, :3]


def _yaw_dominance(command: Tensor, *, like: Tensor) -> Tensor:
    cmd = torch.as_tensor(command, dtype=like.dtype, device=like.device)
    lin = torch.linalg.vector_norm(cmd[:, :2], dim=-1, keepdim=True)
    yaw = torch.abs(cmd[:, 2:3])
    return yaw / torch.clamp(lin + yaw, min=1.0e-6)


def _soft_gate(value: Tensor, *, start: float, span: float) -> Tensor:
    return torch.clamp((value - float(start)) / max(float(span), 1.0e-6), min=0.0, max=1.0)


def _foot_pos_from_body_rel(state: MpcRobotState, rel_body: Tensor) -> Tensor:
    root = torch.as_tensor(state.root_pos, dtype=torch.float32, device=state.foot_pos.device)
    rpy = torch.as_tensor(state.root_rpy, dtype=torch.float32, device=state.foot_pos.device)
    rel_body = torch.as_tensor(rel_body, dtype=torch.float32, device=state.foot_pos.device)
    yaw = rpy[:, 2]
    cy = torch.cos(yaw).unsqueeze(-1)
    sy = torch.sin(yaw).unsqueeze(-1)
    rel_world_xy = torch.stack(
        (
            cy * rel_body[..., 0] - sy * rel_body[..., 1],
            sy * rel_body[..., 0] + cy * rel_body[..., 1],
        ),
        dim=-1,
    )
    rel_world = torch.cat((rel_world_xy, rel_body[..., 2:3]), dim=-1)
    return root.unsqueeze(1) + rel_world


def _memory_seed_state(state: MpcRobotState, command: Tensor, runtime_cfg: MpcRuntimeCfg, memory: MpcFootholdMemory | None) -> MpcRobotState:
    if memory is None or memory.foot_rel_body_seed is None:
        return state
    foot_pos0 = torch.as_tensor(state.foot_pos, dtype=torch.float32, device=state.foot_pos.device)
    linear_dom = 1.0 - _yaw_dominance(command, like=foot_pos0)
    linear_weight = _soft_gate(
        linear_dom,
        start=float(runtime_cfg.foothold_linear_gate_start),
        span=float(runtime_cfg.foothold_linear_gate_span),
    ).to(dtype=foot_pos0.dtype, device=foot_pos0.device)
    seeded_foot = _foot_pos_from_body_rel(state, memory.foot_rel_body_seed).to(dtype=foot_pos0.dtype, device=foot_pos0.device)
    blended_foot = torch.lerp(foot_pos0, seeded_foot, linear_weight[:, None, :])
    return MpcRobotState(
        root_pos=state.root_pos,
        root_rpy=state.root_rpy,
        joint_angles=state.joint_angles,
        foot_pos=blended_foot.to(dtype=state.foot_pos.dtype, device=state.foot_pos.device),
        foot_vel=state.foot_vel,
    )


def _apply_foothold_memory(
    nominal: dict[str, Tensor],
    command: Tensor,
    runtime_cfg: MpcRuntimeCfg,
    memory: MpcFootholdMemory | None,
) -> dict[str, Tensor]:
    if memory is None or memory.stance_anchor_w is None:
        return nominal
    contact = nominal["contact_logits"] > 0.0
    yaw_dom = _yaw_dominance(command, like=nominal["foot_pos"])
    yaw_weight = _soft_gate(
        yaw_dom,
        start=float(runtime_cfg.foothold_yaw_gate_start),
        span=float(runtime_cfg.foothold_yaw_gate_span),
    ).to(dtype=nominal["foot_pos"].dtype, device=nominal["foot_pos"].device)
    if memory.yaw_entry_ramp is not None:
        ramp = torch.as_tensor(memory.yaw_entry_ramp, dtype=yaw_weight.dtype, device=yaw_weight.device).reshape(-1, 1)
        yaw_weight = yaw_weight * ramp
    anchor_t = torch.as_tensor(
        memory.stance_anchor_w,
        dtype=nominal["foot_pos"].dtype,
        device=nominal["foot_pos"].device,
    ).unsqueeze(1)
    yaw_weight_btlf = yaw_weight.view(yaw_weight.shape[0], 1, 1, 1)
    replacement = torch.lerp(nominal["foot_pos"], anchor_t, yaw_weight_btlf)
    nominal["foot_pos"] = torch.where(contact.unsqueeze(-1), replacement, nominal["foot_pos"])

    linear_dom = 1.0 - yaw_dom
    linear_z_weight = _soft_gate(
        linear_dom,
        start=float(runtime_cfg.foothold_linear_gate_start),
        span=float(runtime_cfg.foothold_linear_gate_span),
    ).to(dtype=nominal["foot_pos"].dtype, device=nominal["foot_pos"].device)
    linear_z_weight_btlf = linear_z_weight.view(linear_z_weight.shape[0], 1, 1, 1)
    z_weight = torch.maximum(yaw_weight_btlf, linear_z_weight_btlf)
    grounded_z = torch.lerp(nominal["foot_pos"][..., 2:3], anchor_t[..., 2:3], z_weight)
    nominal["foot_pos"][..., 2:3] = torch.where(contact.unsqueeze(-1), grounded_z, nominal["foot_pos"][..., 2:3])
    return nominal


def build_nominal_trajectory(
    state: MpcRobotState,
    command: Tensor,
    runtime_cfg: MpcRuntimeCfg,
    memory: MpcFootholdMemory | None = None,
) -> dict[str, Tensor]:
    """Build a differentiable nominal seed for root/foot/contact."""
    root_pos0_raw = torch.as_tensor(state.root_pos, dtype=torch.float32)
    cmd = _as_command(command, device=root_pos0_raw.device)
    if bool(runtime_cfg.foothold_memory_enabled):
        state = _memory_seed_state(state, cmd, runtime_cfg, memory)
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

    nominal = {
        "root_pos": root_pos,
        "root_rpy": root_rpy,
        "foot_pos": foot_pos,
        "contact_logits": contact_logits,
    }
    if bool(runtime_cfg.foothold_memory_enabled):
        nominal = _apply_foothold_memory(nominal, cmd, runtime_cfg, memory)
    return nominal


__all__ = ["build_nominal_trajectory"]
