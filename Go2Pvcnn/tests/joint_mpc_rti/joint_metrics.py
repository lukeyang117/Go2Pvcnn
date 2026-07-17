from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class JointMetricTrace:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    command_body: Tensor
    foot_height_w: Tensor
    foot_small_distance_m: Tensor
    part_collision: dict[str, Tensor]
    valid: Tensor
    dt: float


def _command_axis(command_xy: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True)
    fallback = torch.zeros_like(command_xy)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1.0e-6, command_xy / norm.clamp_min(1.0e-6), fallback)


def accumulate_joint_metrics(trace: JointMetricTrace) -> dict[str, float]:
    root = torch.as_tensor(trace.root_pos_w)
    foot = torch.as_tensor(trace.foot_pos_w, dtype=root.dtype, device=root.device)
    contact = torch.as_tensor(trace.contact_state, dtype=torch.bool, device=root.device)
    command = torch.as_tensor(trace.command_body, dtype=root.dtype, device=root.device)
    if root.ndim != 3 or foot.ndim != 4 or contact.shape != foot.shape[:-1]:
        raise ValueError("invalid JointMetricTrace shapes")
    root_step = root[:, 1:, :2] - root[:, :-1, :2]
    foot_step = foot[:, 1:, :, :2] - foot[:, :-1, :, :2]
    axis = _command_axis(command[:, 1:, :2])
    root_progress = (root_step * axis).sum(dim=-1)
    foot_progress = (foot_step * axis.unsqueeze(2)).sum(dim=-1)
    consecutive_stance = torch.logical_and(contact[:, 1:], contact[:, :-1])
    stance_slip = torch.linalg.vector_norm(foot_step, dim=-1)[consecutive_stance]
    if stance_slip.numel():
        stance_max = float(stance_slip.max().item())
        stance_mean = float(stance_slip.mean().item())
        stance_stationary = float((stance_slip <= 0.001).to(root.dtype).mean().item())
        carry = torch.abs(foot_progress / root_progress.unsqueeze(-1).clamp_min(1.0e-8))[consecutive_stance]
        carry_ratio = float(carry.mean().item())
    else:
        stance_max = stance_mean = carry_ratio = 0.0
        stance_stationary = 1.0

    swing = torch.logical_not(contact[:, 1:])
    relative_progress = foot_progress - root_progress.unsqueeze(-1)
    active_denominator = torch.abs(root_progress).unsqueeze(-1).expand_as(relative_progress)
    if swing.any():
        active_ratio = float(
            (relative_progress[swing] / active_denominator[swing].clamp_min(1.0e-8)).mean().item()
        )
    else:
        active_ratio = 0.0

    relative_xy = foot[..., :2] - root[..., None, :2]
    relative_from_start = relative_xy - relative_xy[:, :1]
    foot_onset_distance = (relative_from_start * _command_axis(command[..., :2]).unsqueeze(2)).sum(dim=-1)
    root_from_start = root[..., :2] - root[:, :1, :2]
    root_onset_distance = (root_from_start * _command_axis(command[..., :2])).sum(dim=-1)
    foot_onsets = torch.logical_and(torch.logical_not(contact), foot_onset_distance >= 0.001).any(dim=2)
    root_onsets = root_onset_distance >= 0.0005
    lead_values: list[float] = []
    leak_values: list[float] = []
    for row in range(int(root.shape[0])):
        foot_indices = torch.where(foot_onsets[row])[0]
        root_indices = torch.where(root_onsets[row])[0]
        if foot_indices.numel() == 0 or root_indices.numel() == 0:
            lead_values.append(-float(trace.dt) * 1000.0)
            leak_values.append(float(root_onset_distance[row].clamp_min(0.0).max().item()))
            continue
        foot_index = int(foot_indices[0].item())
        root_index = int(root_indices[0].item())
        lead_values.append(float(root_index - foot_index) * float(trace.dt) * 1000.0)
        leak_values.append(float(root_onset_distance[row, foot_index].clamp_min(0.0).item()))

    output = {
        "stance_xy_slip_max_m": stance_max,
        "stance_xy_slip_mean_m": stance_mean,
        "stance_stationary_ratio": stance_stationary,
        "stance_root_carry_ratio_abs": carry_ratio,
        "swing_active_motion_ratio": active_ratio,
        "foot_root_lead_time_min_ms": min(lead_values),
        "foot_root_lead_time_max_ms": max(lead_values),
        "root_leak_before_foot_m": max(leak_values),
    }
    for part, collision in trace.part_collision.items():
        collision_tensor = torch.as_tensor(collision, dtype=torch.bool, device=root.device)
        output[f"{part}_collision_frame_rate"] = float(collision_tensor.to(root.dtype).mean().item())
    return output


__all__ = ["JointMetricTrace", "accumulate_joint_metrics"]
