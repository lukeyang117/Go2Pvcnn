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
    reliable_stance: Tensor | None = None
    touchdown_ready: Tensor | None = None
    swing_extension_age: Tensor | None = None
    recovery_state: Tensor | None = None
    liftoff_blocked: Tensor | None = None
    root_control: Tensor | None = None
    joint_pos: Tensor | None = None
    line_search_alpha: Tensor | None = None
    line_search_rejection_reason: Tensor | None = None
    root_nominal_pos_w: Tensor | None = None
    root_nominal_rpy_w: Tensor | None = None


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
        stance_root_progress = torch.abs(root_progress).unsqueeze(-1).expand_as(foot_progress)
        carry_ratio = float(
            torch.abs(foot_progress[consecutive_stance]).sum().div(
                stance_root_progress[consecutive_stance].sum().clamp_min(1.0e-8)
            ).item()
        )
    else:
        stance_max = stance_mean = carry_ratio = 0.0
        stance_stationary = 1.0

    active_ratios: list[Tensor] = []
    for row in range(int(root.shape[0])):
        for leg in range(4):
            liftoff = torch.where(
                torch.logical_and(contact[row, :-1, leg], torch.logical_not(contact[row, 1:, leg]))
            )[0] + 1
            touchdown = torch.where(
                torch.logical_and(torch.logical_not(contact[row, :-1, leg]), contact[row, 1:, leg])
            )[0] + 1
            for start_tensor in liftoff:
                start = int(start_tensor.item())
                endings = touchdown[touchdown > start]
                if endings.numel() == 0:
                    continue
                stop = int(endings[0].item())
                event_axis = _command_axis(command[row, start : start + 1, :2])[0]
                relative_start = foot[row, start - 1, leg, :2] - root[row, start - 1, :2]
                relative_stop = foot[row, stop, leg, :2] - root[row, stop, :2]
                relative_event = ((relative_stop - relative_start) * event_axis).sum()
                expected_progress = (
                    torch.linalg.vector_norm(command[row, start, :2])
                    * float(stop - start + 1)
                    * float(trace.dt)
                )
                active_ratios.append(relative_event / expected_progress.clamp_min(1.0e-8))
    active_ratio = float(torch.stack(active_ratios).mean().item()) if active_ratios else 0.0

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

    rpy = torch.as_tensor(trace.root_rpy_w, dtype=root.dtype, device=root.device)
    valid = torch.as_tensor(trace.valid, dtype=torch.bool, device=root.device)
    valid_rpy = rpy[valid]
    roll_deg = torch.rad2deg(valid_rpy[:, 0]) if valid_rpy.numel() else root.new_zeros(1)
    pitch_deg = torch.rad2deg(valid_rpy[:, 1]) if valid_rpy.numel() else root.new_zeros(1)

    def signed_summary(value: Tensor, prefix: str) -> dict[str, float]:
        absolute = value.abs()
        return {
            f"{prefix}_mean_deg": float(value.mean().item()),
            f"{prefix}_p95_deg": float(torch.quantile(absolute, 0.95).item()),
            f"{prefix}_abs_max_deg": float(absolute.max().item()),
        }

    rpy_rate = torch.diff(rpy, dim=1) / float(trace.dt)
    rate_valid = torch.logical_and(valid[:, 1:], valid[:, :-1])
    valid_rpy_rate = rpy_rate[rate_valid]
    root_roll_pitch_rate_max = (
        float(valid_rpy_rate[:, :2].abs().max().item()) if valid_rpy_rate.numel() else 0.0
    )

    nominal_root = trace.root_nominal_pos_w
    nominal_rpy = trace.root_nominal_rpy_w
    nominal_rpy_tensor = rpy if nominal_rpy is None else torch.as_tensor(
        nominal_rpy, dtype=root.dtype, device=root.device
    )
    if nominal_root is None:
        root_lateral_offset = root.new_zeros(())
        root_lateral_velocity_error = root.new_zeros(())
    else:
        nominal_root = torch.as_tensor(nominal_root, dtype=root.dtype, device=root.device)
        cosine = torch.cos(nominal_rpy_tensor[..., 2])
        sine = torch.sin(nominal_rpy_tensor[..., 2])
        command_world = torch.stack(
            (command[..., 0] * cosine - command[..., 1] * sine,
             command[..., 0] * sine + command[..., 1] * cosine),
            dim=-1,
        )
        axis = command_world / torch.linalg.vector_norm(command_world, dim=-1, keepdim=True).clamp_min(1.0e-6)
        lateral = torch.stack((-axis[..., 1], axis[..., 0]), dim=-1)
        root_lateral_offset = ((root[..., :2] - nominal_root[..., :2]) * lateral).sum(dim=-1).abs().amax()
        actual_velocity = torch.diff(root[..., :2], dim=1) / float(trace.dt)
        nominal_velocity = torch.diff(nominal_root[..., :2], dim=1) / float(trace.dt)
        root_lateral_velocity_error = ((actual_velocity - nominal_velocity) * lateral[:, 1:]).sum(dim=-1).abs().amax()
    if nominal_rpy is None:
        root_yaw_error = root.new_zeros(())
        root_yaw_rate_assist = root.new_zeros(())
    else:
        nominal_rpy = nominal_rpy_tensor
        yaw_delta = torch.atan2(torch.sin(rpy[..., 2] - nominal_rpy[..., 2]), torch.cos(rpy[..., 2] - nominal_rpy[..., 2]))
        root_yaw_error = torch.rad2deg(yaw_delta).abs().amax()
        root_yaw_rate_assist = torch.diff(yaw_delta, dim=1).abs().amax() / float(trace.dt)

    touchdown = torch.zeros_like(contact)
    touchdown[:, 1:] = torch.logical_and(torch.logical_not(contact[:, :-1]), contact[:, 1:])
    reliable = contact if trace.reliable_stance is None else torch.as_tensor(
        trace.reliable_stance, dtype=torch.bool, device=root.device
    )
    touchdown_count = int(touchdown.sum().item())
    unsafe_anchor_count = int(torch.logical_and(touchdown, torch.logical_not(reliable)).sum().item())
    foot_surface_error = foot[..., 2] - torch.as_tensor(
        trace.foot_height_w, dtype=root.dtype, device=root.device
    ) - 0.022
    airborne_touchdown_count = int(torch.logical_and(touchdown, foot_surface_error > 0.010).sum().item())
    touchdown_on_small_count = int(
        torch.logical_and(
            touchdown,
            torch.as_tensor(trace.foot_small_distance_m, dtype=root.dtype, device=root.device) < 0.052,
        ).sum().item()
    )

    extension = (
        torch.zeros_like(contact, dtype=torch.long)
        if trace.swing_extension_age is None
        else torch.as_tensor(trace.swing_extension_age, dtype=torch.long, device=root.device)
    )
    recovery = (
        torch.zeros_like(contact)
        if trace.recovery_state is None
        else torch.as_tensor(trace.recovery_state, dtype=torch.bool, device=root.device)
    )
    extension_before_touchdown = torch.zeros_like(extension)
    extension_before_touchdown[:, 1:] = extension[:, :-1]
    forced_touchdown = torch.logical_and(
        touchdown,
        torch.logical_and(extension_before_touchdown >= 10, torch.logical_not(reliable)),
    )
    liftoff_blocked = (
        torch.zeros_like(contact)
        if trace.liftoff_blocked is None
        else torch.as_tensor(trace.liftoff_blocked, dtype=torch.bool, device=root.device)
    )
    liftoff = torch.zeros_like(contact)
    liftoff[:, 1:] = torch.logical_and(contact[:, :-1], torch.logical_not(contact[:, 1:]))
    liftoff_guard_violation = torch.logical_and(liftoff, liftoff_blocked)
    safe_support_count = reliable.to(torch.long).sum(dim=2)
    zero_support = safe_support_count == 0
    zero_support_run = 0
    for row in range(int(zero_support.shape[0])):
        run = 0
        for value in zero_support[row]:
            run = run + 1 if bool(value) else 0
            zero_support_run = max(zero_support_run, run)

    output = {
        "stance_xy_slip_max_m": stance_max,
        "stance_xy_slip_mean_m": stance_mean,
        "stance_stationary_ratio": stance_stationary,
        "stance_root_carry_ratio_abs": carry_ratio,
        "swing_active_motion_ratio": active_ratio,
        "foot_root_lead_time_min_ms": min(lead_values),
        "foot_root_lead_time_max_ms": max(lead_values),
        "root_leak_before_foot_m": max(leak_values),
        **signed_summary(roll_deg, "root_roll_error"),
        **signed_summary(pitch_deg, "root_pitch_error"),
        "root_roll_pitch_rate_max_rps": root_roll_pitch_rate_max,
        "root_lateral_offset_from_nominal_m": float(root_lateral_offset.item()),
        "root_lateral_velocity_error_mps": float(root_lateral_velocity_error.item()),
        "root_yaw_error_from_nominal_deg": float(root_yaw_error.item()),
        "root_yaw_rate_assist_error_rps": float(root_yaw_rate_assist.item()),
        "scheduled_touchdown_count": touchdown_count,
        "confirmed_touchdown_count": touchdown_count,
        "airborne_touchdown_count": airborne_touchdown_count,
        "touchdown_on_small_count": touchdown_on_small_count,
        "unsafe_stance_anchor_count": unsafe_anchor_count,
        "swing_extension_frames_max": int(extension.max().item()),
        "recovery_frame_count": int(recovery.sum().item()),
        "forced_touchdown_after_extension_count": int(forced_touchdown.sum().item()),
        "liftoff_blocked_count": int(liftoff_blocked.sum().item()),
        "liftoff_guard_violation_count": int(liftoff_guard_violation.sum().item()),
        "safe_support_count_min": int(safe_support_count.min().item()),
        "zero_support_run": zero_support_run,
    }
    for part, collision in trace.part_collision.items():
        collision_tensor = torch.as_tensor(collision, dtype=torch.bool, device=root.device)
        output[f"{part}_collision_frame_rate"] = float(collision_tensor.to(root.dtype).mean().item())
        output[f"{part}_collision_frames"] = int(collision_tensor.sum().item())
    if trace.line_search_alpha is not None:
        alpha = torch.as_tensor(trace.line_search_alpha, dtype=root.dtype, device=root.device)
        for value, label in ((1.0, "1"), (0.5, "0_5"), (0.25, "0_25"), (0.1, "0_1"), (0.0, "0")):
            selected = torch.isclose(alpha, alpha.new_tensor(value))
            output[f"line_search_alpha_{label}_count"] = int(selected.sum().item())
        zero = torch.isclose(alpha, alpha.new_tensor(0.0))
        max_run = 0
        for row in range(int(zero.shape[0])):
            run = 0
            for value in zero[row].reshape(-1):
                run = run + 1 if bool(value) else 0
                max_run = max(max_run, run)
        output["line_search_alpha_0_max_run"] = max_run
        output["line_search_alpha_0_rate"] = float(zero.to(root.dtype).mean().item())
    return output


__all__ = ["JointMetricTrace", "accumulate_joint_metrics"]
