"""Shared trace schema and metrics for flat and small-obstacle acceptance."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


JOINT_LOWER = torch.tensor((-1.0472, -0.6632, -2.721) * 4)
JOINT_UPPER = torch.tensor((1.0472, 2.966, -0.837) * 4)
PARTS = ("foot", "knee", "calf", "thigh", "base")

FLAT_METRICS = frozenset(
    {
        "joint_position_violation",
        "joint_velocity_violation",
        "joint_safe_margin_min_rad",
        "joint_step_max_rad",
        "joint_acceleration_max_rps2",
        "stance_ground_gap",
        "stance_ground_penetration",
        "stance_anchor_residual",
        "stance_xy_slip_max_m",
        "stance_xy_slip_mean_m",
        "stance_stationary_ratio",
        "stance_root_carry_ratio_abs",
        "swing_surface_clearance_min_m",
        "swing_active_motion_ratio",
        "foot_root_lead_time_min_ms",
        "foot_root_lead_time_max_ms",
        "root_leak_before_foot_m",
        "root_velocity_error",
        "root_direction_error",
        "root_yaw_rate_error",
        "root_zero_drift_m",
        "root_step_jump_m",
        "root_roll_deviation_rad",
        "root_pitch_deviation_rad",
        "line_alpha_zero_ratio",
        "line_alpha_zero_run",
        "warm_start_jump_max",
        "trajectory_valid_ratio",
        "nonfinite_count",
        "x0_injection_error",
        "published_x1_error",
        "map_valid_ratio",
        "cold_start_count",
        "warm_start_count",
        "unexpected_cold_restart_count",
        "warm_cache_invariant_fault_count",
    }
)
SMALL_METRICS = frozenset(
    {
        "strict_cross_success",
        "touchdown_on_small",
        "stance_on_small",
        "airborne_touchdown",
        "root_lateral_offset_from_nominal_m",
        "root_lateral_velocity_error_mps",
        "root_yaw_error_from_nominal_rad",
        "root_roll_pitch_rate_max_rps",
        "maximum_penetration_m",
        *(f"{part}_collision_frame_rate" for part in PARTS),
    }
)


@dataclass(frozen=True)
class JointMetricTrace:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    joint_pos: Tensor
    foot_pos_w: Tensor
    contact_state: Tensor
    command_body: Tensor
    gait_phase: Tensor
    foot_height_w: Tensor
    foot_small_distance_m: Tensor
    part_collision: dict[str, Tensor]
    line_alpha: Tensor
    nominal_root_pos_w: Tensor
    nominal_root_rpy_w: Tensor
    valid: Tensor
    map_valid: Tensor
    timestamps: Tensor
    dt: float
    stance_anchor_w: Tensor | None = None
    strict_cross_success: Tensor | None = None
    touchdown_on_small: Tensor | None = None
    stance_on_small: Tensor | None = None
    airborne_touchdown: Tensor | None = None
    part_penetration_m: dict[str, Tensor] | None = None
    x0_injection_error: Tensor | None = None
    published_x1_error: Tensor | None = None
    warm_start_jump: Tensor | None = None
    cold_start: Tensor | None = None
    warm_start: Tensor | None = None
    warm_cache_invariant_fault: Tensor | None = None


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float | int | None
    numerator: float | int | None
    denominator: float | int | None
    valid_count: int
    applicable: bool
    na_reason: str | None
    threshold: float | None
    passed: bool | None
    worst_case_key: tuple[str, ...] | None


@dataclass(frozen=True)
class JointMetricReport:
    scenario: str
    metrics: dict[str, MetricResult]

    def metric(self, name: str) -> MetricResult:
        return self.metrics[name]

    @property
    def passed(self) -> bool:
        return all(metric.passed is not False for metric in self.metrics.values() if metric.applicable)


def _command_axis(command_xy: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True)
    fallback = torch.zeros_like(command_xy)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1.0e-6, command_xy / norm.clamp_min(1.0e-6), fallback)


def _world_command_axis(command_xy: Tensor, yaw_w: Tensor) -> Tensor:
    axis_body = _command_axis(command_xy)
    yaw = torch.as_tensor(yaw_w, dtype=axis_body.dtype, device=axis_body.device)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * axis_body[..., 0] - sine * axis_body[..., 1],
            sine * axis_body[..., 0] + cosine * axis_body[..., 1],
        ),
        dim=-1,
    )


def _masked_max(value: Tensor, mask: Tensor, default: float = 0.0) -> float:
    selected = value[mask]
    return default if selected.numel() == 0 else float(selected.max().item())


def _masked_mean(value: Tensor, mask: Tensor, default: float = 0.0) -> float:
    selected = value[mask]
    return default if selected.numel() == 0 else float(selected.mean().item())


def _masked_bool_ratio(value: Tensor, mask: Tensor, default: float = 0.0) -> float:
    selected = value[mask].to(dtype=torch.int64)
    if selected.numel() == 0:
        return default
    return float(selected.sum().item()) / float(selected.numel())


def _max_true_run(mask: Tensor) -> int:
    maximum = 0
    current = torch.zeros(mask.shape[0], dtype=torch.long, device=mask.device)
    for index in range(mask.shape[1]):
        current = torch.where(mask[:, index], current + 1, torch.zeros_like(current))
        maximum = max(maximum, int(current.max().item()))
    return maximum


def applicable_metrics(scenario: str, command: tuple[float, float, float] | None = None) -> frozenset[str]:
    if scenario not in ("flat", "small"):
        raise ValueError("scenario must be flat or small")
    metrics = set(FLAT_METRICS if scenario == "flat" else FLAT_METRICS | SMALL_METRICS)
    if command is not None and math.hypot(command[0], command[1]) <= 1.0e-8:
        metrics.discard("strict_cross_success")
        metrics.discard("root_direction_error")
        metrics.discard("swing_active_motion_ratio")
        metrics.discard("foot_root_lead_time_min_ms")
        metrics.discard("foot_root_lead_time_max_ms")
        metrics.discard("root_leak_before_foot_m")
        metrics.discard("stance_root_carry_ratio_abs")
    else:
        metrics.discard("root_zero_drift_m")
    return frozenset(metrics)


def accumulate_joint_metrics(trace: JointMetricTrace) -> dict[str, float | int]:
    root = torch.as_tensor(trace.root_pos_w)
    rpy = torch.as_tensor(trace.root_rpy_w, dtype=root.dtype, device=root.device)
    joint = torch.as_tensor(trace.joint_pos, dtype=root.dtype, device=root.device)
    foot = torch.as_tensor(trace.foot_pos_w, dtype=root.dtype, device=root.device)
    contact = torch.as_tensor(trace.contact_state, dtype=torch.bool, device=root.device)
    command = torch.as_tensor(trace.command_body, dtype=root.dtype, device=root.device)
    valid = torch.as_tensor(trace.valid, dtype=torch.bool, device=root.device)
    future_node = valid & (torch.as_tensor(trace.timestamps, device=root.device) > 0.0)
    if root.ndim != 3 or joint.shape[:2] != root.shape[:2] or foot.shape[:2] != root.shape[:2]:
        raise ValueError("invalid JointMetricTrace shapes")
    valid_edge = valid[:, 1:] & valid[:, :-1]
    root_step = root[:, 1:, :2] - root[:, :-1, :2]
    joint_step = joint[:, 1:] - joint[:, :-1]
    foot_step = foot[:, 1:, :, :2] - foot[:, :-1, :, :2]
    command_edge = command[:, 1:]
    axis = _world_command_axis(command_edge[..., :2], rpy[:, :-1, 2])
    root_progress = (root_step * axis).sum(dim=-1)
    foot_progress = (foot_step * axis.unsqueeze(2)).sum(dim=-1)
    consecutive_stance = contact[:, 1:] & contact[:, :-1] & valid_edge[..., None]
    stance_slip = torch.linalg.vector_norm(foot_step, dim=-1)
    root_denominator = torch.linalg.vector_norm(root_step, dim=-1).unsqueeze(-1).clamp_min(1.0e-8)
    carry = torch.linalg.vector_norm(foot_step, dim=-1) / root_denominator
    relative_progress = foot_progress - root_progress.unsqueeze(-1)
    swing = ~contact[:, 1:] & valid[:, 1:, None]

    relative_xy = foot[..., :2] - root[..., None, :2]
    relative_from_start = relative_xy - relative_xy[:, :1]
    node_axis = _world_command_axis(command[..., :2], rpy[..., 2])
    foot_onset_distance = (relative_from_start * node_axis.unsqueeze(2)).sum(dim=-1)
    root_from_start = root[..., :2] - root[:, :1, :2]
    root_onset_distance = (root_from_start * node_axis).sum(dim=-1)
    foot_onsets = ((~contact) & (foot_onset_distance >= 0.001)).any(dim=2)
    root_onsets = root_onset_distance >= 0.0005
    lead_values: list[float] = []
    leak_values: list[float] = []
    for row in range(root.shape[0]):
        foot_indices = torch.where(foot_onsets[row] & valid[row])[0]
        root_indices = torch.where(root_onsets[row] & valid[row])[0]
        if foot_indices.numel() and root_indices.numel():
            foot_index = int(foot_indices[0].item())
            root_index = int(root_indices[0].item())
            lead_values.append(float(root_index - foot_index) * float(trace.dt) * 1000.0)
            leak_values.append(float(root_onset_distance[row, foot_index].clamp_min(0.0).item()))

    lower = JOINT_LOWER.to(joint)
    upper = JOINT_UPPER.to(joint)
    velocity = joint_step / float(trace.dt)
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / float(trace.dt)
    root_velocity = root_step / float(trace.dt)
    yaw0 = rpy[:, :-1, 2]
    root_velocity_body = torch.stack(
        (
            torch.cos(yaw0) * root_velocity[..., 0] + torch.sin(yaw0) * root_velocity[..., 1],
            -torch.sin(yaw0) * root_velocity[..., 0] + torch.cos(yaw0) * root_velocity[..., 1],
        ),
        dim=-1,
    )
    velocity_error = torch.linalg.vector_norm(root_velocity_body - command_edge[..., :2], dim=-1)
    yaw_rate = (rpy[:, 1:, 2] - rpy[:, :-1, 2]) / float(trace.dt)
    yaw_error = torch.abs(yaw_rate - command_edge[..., 2])
    movement_norm = torch.linalg.vector_norm(root_step, dim=-1)
    command_norm = torch.linalg.vector_norm(command_edge[..., :2], dim=-1)
    cosine = (root_velocity_body * command_edge[..., :2]).sum(dim=-1) / (
        torch.linalg.vector_norm(root_velocity_body, dim=-1) * command_norm
    ).clamp_min(1.0e-8)
    direction_error = torch.acos(cosine.clamp(-1.0, 1.0))
    stance_surface_error = foot[..., 2] - torch.as_tensor(trace.foot_height_w, dtype=root.dtype, device=root.device) - 0.022
    stance_mask = contact & future_node[..., None]
    swing_clearance = foot[..., 2] - torch.as_tensor(trace.foot_height_w, dtype=root.dtype, device=root.device) - 0.022
    nominal_pos = torch.as_tensor(trace.nominal_root_pos_w, dtype=root.dtype, device=root.device)
    nominal_rpy = torch.as_tensor(trace.nominal_root_rpy_w, dtype=root.dtype, device=root.device)
    nominal_axis = _world_command_axis(command[..., :2], nominal_rpy[..., 2])
    lateral_axis = torch.stack((-nominal_axis[..., 1], nominal_axis[..., 0]), dim=-1)
    lateral = ((root[..., :2] - nominal_pos[..., :2]) * lateral_axis).sum(dim=-1)

    output: dict[str, float | int] = {
        "trajectory_valid_ratio": _masked_bool_ratio(valid, torch.ones_like(valid, dtype=torch.bool)),
        "joint_position_violation": int((((joint < lower) | (joint > upper)) & valid[..., None]).sum().item()),
        "joint_velocity_violation": int(((velocity.abs() > 30.0) & valid_edge[..., None]).sum().item()),
        "joint_safe_margin_min_rad": float(
            torch.where(valid[..., None], torch.minimum(joint - lower, upper - joint), torch.full_like(joint, torch.inf)).min().item()
        ),
        "joint_step_max_rad": _masked_max(joint_step.abs().amax(dim=-1), valid_edge),
        "joint_acceleration_max_rps2": _masked_max(acceleration.abs().amax(dim=-1), valid_edge[:, 1:] & valid_edge[:, :-1]),
        "stance_xy_slip_max_m": _masked_max(stance_slip, consecutive_stance),
        "stance_xy_slip_mean_m": _masked_mean(stance_slip, consecutive_stance),
        "stance_stationary_ratio": _masked_bool_ratio(
            stance_slip <= 0.0005, consecutive_stance, 1.0
        ),
        "stance_root_carry_ratio_abs": _masked_mean(carry, consecutive_stance),
        "swing_active_motion_ratio": _masked_mean(relative_progress / root_denominator, swing),
        "foot_root_lead_time_min_ms": min(lead_values) if lead_values else 0.0,
        "foot_root_lead_time_max_ms": max(lead_values) if lead_values else 0.0,
        "root_leak_before_foot_m": max(leak_values) if leak_values else 0.0,
        "root_velocity_error": _masked_mean(velocity_error, valid_edge),
        "root_direction_error": _masked_mean(direction_error, valid_edge & (command_norm > 1.0e-6)),
        "root_yaw_rate_error": _masked_mean(yaw_error, valid_edge),
        "root_zero_drift_m": float(torch.linalg.vector_norm(root[:, -1, :2] - root[:, 0, :2], dim=-1).max().item()),
        "root_step_jump_m": _masked_max(torch.linalg.vector_norm(root[:, 1:] - root[:, :-1], dim=-1), valid_edge),
        "root_roll_deviation_rad": _masked_max(torch.abs(rpy[..., 0] - nominal_rpy[..., 0]), valid),
        "root_pitch_deviation_rad": _masked_max(torch.abs(rpy[..., 1] - nominal_rpy[..., 1]), valid),
        "root_lateral_offset_from_nominal_m": _masked_max(torch.abs(lateral), valid),
        "root_lateral_velocity_error_mps": _masked_max(torch.abs(lateral[:, 1:] - lateral[:, :-1]) / float(trace.dt), valid_edge),
        "root_yaw_error_from_nominal_rad": _masked_max(torch.abs(rpy[..., 2] - nominal_rpy[..., 2]), valid),
        "root_roll_pitch_rate_max_rps": _masked_max(torch.abs((rpy[:, 1:, :2] - rpy[:, :-1, :2]) / float(trace.dt)).amax(dim=-1), valid_edge),
        "stance_ground_gap": _masked_max(torch.clamp_min(stance_surface_error, 0.0), stance_mask),
        "stance_ground_penetration": _masked_max(torch.clamp_min(-stance_surface_error, 0.0), stance_mask),
        "swing_surface_clearance_min_m": -_masked_max(-swing_clearance, (~contact) & future_node[..., None]),
        "line_alpha_zero_ratio": _masked_mean((torch.as_tensor(trace.line_alpha, device=root.device) == 0).to(root.dtype), valid),
        "line_alpha_zero_run": _max_true_run((torch.as_tensor(trace.line_alpha, device=root.device) == 0) & valid),
        "nonfinite_count": int((~torch.isfinite(root)).sum().item() + (~torch.isfinite(joint)).sum().item() + (~torch.isfinite(foot)).sum().item()),
        "map_valid_ratio": _masked_bool_ratio(
            torch.as_tensor(trace.map_valid, dtype=torch.bool, device=root.device), valid
        ),
        "cold_start_count": 1 if trace.cold_start is None else int(
            torch.as_tensor(trace.cold_start, dtype=torch.bool, device=root.device).sum().item()
        ),
        "warm_start_count": 0 if trace.warm_start is None else int(
            torch.as_tensor(trace.warm_start, dtype=torch.bool, device=root.device).sum().item()
        ),
        "unexpected_cold_restart_count": 0,
        "warm_cache_invariant_fault_count": 0 if trace.warm_cache_invariant_fault is None else int(
            torch.as_tensor(trace.warm_cache_invariant_fault, dtype=torch.bool, device=root.device).sum().item()
        ),
    }
    if trace.cold_start is not None:
        cold = torch.as_tensor(trace.cold_start, dtype=torch.bool, device=root.device)
        cold_count = cold.to(torch.int64).sum(dim=1)
        output["unexpected_cold_restart_count"] = int((cold_count - 1).clamp_min(0).sum().item())
    anchor = trace.stance_anchor_w
    output["stance_anchor_residual"] = 0.0 if anchor is None else _masked_max(
        torch.linalg.vector_norm(
            foot[..., :2] - torch.as_tensor(anchor, dtype=root.dtype, device=root.device)[..., :2],
            dim=-1,
        ),
        stance_mask,
    )
    for name, tensor in (
        ("x0_injection_error", trace.x0_injection_error),
        ("published_x1_error", trace.published_x1_error),
        ("warm_start_jump_max", trace.warm_start_jump),
    ):
        output[name] = 0.0 if tensor is None else float(torch.as_tensor(tensor).abs().max().item())
    for name, tensor in (
        ("strict_cross_success", trace.strict_cross_success),
        ("touchdown_on_small", trace.touchdown_on_small),
        ("stance_on_small", trace.stance_on_small),
        ("airborne_touchdown", trace.airborne_touchdown),
    ):
        output[name] = 0.0 if tensor is None else float(torch.as_tensor(tensor, dtype=root.dtype).mean().item())
    penetration_values: list[Tensor] = []
    for part in PARTS:
        collision = trace.part_collision.get(part)
        output[f"{part}_collision_frame_rate"] = 0.0 if collision is None else float(
            torch.as_tensor(collision, dtype=root.dtype).mean().item()
        )
        if trace.part_penetration_m and part in trace.part_penetration_m:
            penetration_values.append(torch.as_tensor(trace.part_penetration_m[part], dtype=root.dtype))
    output["maximum_penetration_m"] = 0.0 if not penetration_values else float(
        torch.stack(tuple(value.max() for value in penetration_values)).max().item()
    )
    return output


def evaluate_trace(
    trace: JointMetricTrace,
    *,
    scenario: str,
    key: tuple[str, ...] | None = None,
) -> JointMetricReport:
    from .acceptance_thresholds import THRESHOLDS

    command0 = tuple(float(value) for value in torch.as_tensor(trace.command_body)[0, 0].tolist())
    applicable = applicable_metrics(scenario, command0)
    values = accumulate_joint_metrics(trace)
    names = FLAT_METRICS | SMALL_METRICS
    valid_count = int(torch.as_tensor(trace.valid, dtype=torch.bool).sum().item())
    metrics: dict[str, MetricResult] = {}
    for name in names:
        is_applicable = name in applicable
        if not is_applicable:
            reason = "no small obstacle in flat scenario" if name in SMALL_METRICS else "zero translation command"
        else:
            reason = None
        value = values.get(name)
        operator, threshold = THRESHOLDS.get(name, (None, None))
        if not is_applicable or value is None:
            passed = None
        elif operator == "le":
            passed = float(value) <= float(threshold)
        elif operator == "ge":
            passed = float(value) >= float(threshold)
        else:
            passed = True
        metrics[name] = MetricResult(
            name=name,
            value=value,
            numerator=value,
            denominator=valid_count,
            valid_count=valid_count,
            applicable=is_applicable,
            na_reason=reason,
            threshold=threshold,
            passed=passed,
            worst_case_key=key,
        )
    return JointMetricReport(scenario=scenario, metrics=metrics)


__all__ = [
    "FLAT_METRICS",
    "SMALL_METRICS",
    "JointMetricReport",
    "JointMetricTrace",
    "MetricResult",
    "accumulate_joint_metrics",
    "applicable_metrics",
    "evaluate_trace",
]
