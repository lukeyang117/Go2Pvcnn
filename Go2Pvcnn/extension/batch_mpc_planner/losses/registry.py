"""Loss registry for batch MPC."""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import MpcPlannerCfg
from ..variables import DecodedMpcTrajectory
from .contact import contact_binary_loss, contact_schedule_tracking_loss, contact_transition_loss, support_stability_loss
from .gait_coupling import root_frame_drift_loss, root_frame_follow_loss, stance_slip_loss, swing_stride_loss
from .kinematics import joint_limit_loss
from .smoothness import foot_smoothness_loss, root_smoothness_loss
from .terrain_clearance import obstacle_margin_loss, swing_clearance_loss, terrain_clearance_loss
from .tracking import command_tracking_loss, progress_direction_loss


def _weighted(enabled: bool, weight: float | Tensor, value: Tensor, breakdown: dict[str, Tensor], name: str) -> Tensor:
    if not enabled:
        zero = torch.zeros_like(value)
        breakdown[name] = zero
        return zero
    if isinstance(weight, Tensor):
        out = weight.to(dtype=value.dtype, device=value.device) * value
    else:
        out = float(weight) * value
    breakdown[name] = out
    return out


def _command_adaptive_weights(command: Tensor, *, like: Tensor) -> dict[str, Tensor]:
    cmd = torch.as_tensor(command, dtype=like.dtype, device=like.device)
    if cmd.ndim != 2 or int(cmd.shape[1]) < 3:
        zeros = torch.zeros((like.shape[0],), dtype=like.dtype, device=like.device)
        ones = torch.ones_like(zeros)
        return {
            "motion": ones,
            "swing": ones,
            "follow": ones,
        }
    lin_speed = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
    abs_vx = torch.abs(cmd[:, 0])
    abs_vy = torch.abs(cmd[:, 1])
    abs_w = torch.abs(cmd[:, 2])
    eps = torch.full_like(lin_speed, 1.0e-6)
    lateral_ratio = abs_vy / torch.clamp(lin_speed, min=eps)
    forward_ratio = abs_vx / torch.clamp(lin_speed, min=eps)
    yaw_ratio = abs_w / (lin_speed + abs_w + eps)
    motion = 1.0 + 1.50 * lateral_ratio + 1.00 * yaw_ratio
    swing = 1.0 + 0.75 * forward_ratio + 0.75 * lateral_ratio + 0.50 * yaw_ratio
    follow = 1.0 + 2.00 * lateral_ratio + 1.50 * yaw_ratio
    return {
        "motion": motion,
        "swing": swing,
        "follow": follow,
    }


def compute_total_loss(
    decoded: DecodedMpcTrajectory,
    nominal: dict[str, Tensor],
    joint_angles: Tensor,
    command: Tensor,
    cfg: MpcPlannerCfg,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return scalar loss, per-env loss, and term breakdown."""
    losses = cfg.losses
    runtime = cfg.runtime
    breakdown: dict[str, Tensor] = {}
    per_env = torch.zeros(decoded.root_pos.shape[0], dtype=decoded.root_pos.dtype, device=decoded.root_pos.device)
    cmd_weights = _command_adaptive_weights(command, like=per_env)

    track = command_tracking_loss(decoded.root_pos, decoded.root_rpy, command, runtime.dt)
    per_env = per_env + _weighted(losses.tracking.enabled, losses.tracking.weight, track, breakdown, "tracking")

    root_s = root_smoothness_loss(decoded.root_pos, decoded.root_rpy)
    foot_s = foot_smoothness_loss(decoded.foot_pos)
    smooth = losses.smoothness.root_weight * root_s + losses.smoothness.foot_weight * foot_s
    per_env = per_env + _weighted(losses.smoothness.enabled, losses.smoothness.weight, smooth, breakdown, "smoothness")

    cbin = contact_binary_loss(decoded.contact_prob)
    ctran = contact_transition_loss(decoded.contact_prob)
    support = support_stability_loss(
        decoded.contact_prob,
        min_support_legs=losses.contact_regularization.min_support_legs,
    )
    contact_loss = (
        losses.contact_regularization.binary_weight * cbin
        + losses.contact_regularization.transition_weight * ctran
        + support
    )
    contact_reg_weight = losses.contact_regularization.weight * cmd_weights["motion"]
    per_env = per_env + _weighted(
        losses.contact_regularization.enabled,
        contact_reg_weight,
        contact_loss,
        breakdown,
        "contact_regularization",
    )
    nominal_contact_prob = torch.sigmoid(nominal["contact_logits"] / float(runtime.contact_temperature))
    contact_schedule = contact_schedule_tracking_loss(
        decoded.contact_prob,
        nominal_contact_prob,
        min_support_prob=losses.contact_schedule.min_support_prob,
    )
    contact_sched_weight = losses.contact_schedule.weight * cmd_weights["motion"]
    per_env = per_env + _weighted(
        losses.contact_schedule.enabled,
        contact_sched_weight,
        contact_schedule,
        breakdown,
        "contact_schedule",
    )

    swing = swing_clearance_loss(decoded.foot_pos, decoded.contact_prob, losses.swing_clearance.min_clearance_m)
    swing_weight = losses.swing_clearance.weight * cmd_weights["swing"]
    per_env = per_env + _weighted(losses.swing_clearance.enabled, swing_weight, swing, breakdown, "swing_clearance")

    stance_slip = stance_slip_loss(
        decoded.contact_prob,
        decoded.foot_pos,
        slip_tolerance_m_per_step=losses.stance_slip.slip_tolerance_m_per_step,
    )
    stance_weight = losses.stance_slip.weight * cmd_weights["motion"]
    per_env = per_env + _weighted(losses.stance_slip.enabled, stance_weight, stance_slip, breakdown, "stance_slip")

    swing_stride = swing_stride_loss(
        decoded.contact_prob,
        decoded.foot_pos,
        command,
        min_swing_span_m=losses.swing_stride.min_swing_span_m,
        command_speed_deadzone_mps=losses.swing_stride.command_speed_deadzone_mps,
    )
    swing_stride_weight = losses.swing_stride.weight * cmd_weights["swing"]
    per_env = per_env + _weighted(losses.swing_stride.enabled, swing_stride_weight, swing_stride, breakdown, "swing_stride")

    root_drift = root_frame_drift_loss(
        decoded.root_pos,
        decoded.foot_pos,
        min_rel_m=losses.root_frame_drift.min_rel_m,
        max_rel_m=losses.root_frame_drift.max_rel_m,
    )
    root_drift_weight = losses.root_frame_drift.weight * cmd_weights["follow"]
    per_env = per_env + _weighted(losses.root_frame_drift.enabled, root_drift_weight, root_drift, breakdown, "root_frame_drift")

    root_follow = root_frame_follow_loss(
        decoded.root_pos,
        decoded.foot_pos,
        rel_change_tolerance_m_per_step=losses.root_frame_follow.rel_change_tolerance_m_per_step,
    )
    root_follow_weight = losses.root_frame_follow.weight * cmd_weights["follow"]
    per_env = per_env + _weighted(losses.root_frame_follow.enabled, root_follow_weight, root_follow, breakdown, "root_frame_follow")

    terrain = terrain_clearance_loss(decoded.foot_pos, losses.terrain_clearance.min_clearance_m)
    per_env = per_env + _weighted(losses.terrain_clearance.enabled, losses.terrain_clearance.weight, terrain, breakdown, "terrain_clearance")

    small_obs = obstacle_margin_loss(
        decoded.foot_pos,
        losses.obstacle_small.body_margin_m,
        losses.obstacle_small.foot_margin_m,
    )
    per_env = per_env + _weighted(losses.obstacle_small.enabled, losses.obstacle_small.weight, small_obs, breakdown, "obstacle_small")

    large_obs = obstacle_margin_loss(
        decoded.foot_pos,
        losses.obstacle_large.body_margin_m,
        losses.obstacle_large.foot_margin_m,
    )
    per_env = per_env + _weighted(losses.obstacle_large.enabled, losses.obstacle_large.weight, large_obs, breakdown, "obstacle_large")

    progress = progress_direction_loss(decoded.root_pos, command, losses.progress.min_progress_m)
    per_env = per_env + _weighted(losses.progress.enabled, losses.progress.weight, progress, breakdown, "progress")

    kin = joint_limit_loss(
        joint_angles,
        joint_limit_rad=losses.kinematics.joint_limit_rad,
        joint_limit_margin_rad=losses.kinematics.joint_limit_margin_rad,
    )
    per_env = per_env + _weighted(losses.kinematics.enabled, losses.kinematics.weight, kin, breakdown, "kinematics")

    touchdown = torch.linalg.norm(decoded.foot_pos[:, :, :, :] - nominal["foot_pos"], dim=-1).mean(dim=(1, 2))
    per_env = per_env + _weighted(losses.touchdown_support.enabled, losses.touchdown_support.weight, touchdown, breakdown, "touchdown_support")

    per_env = torch.nan_to_num(per_env, nan=1e6, posinf=1e6, neginf=1e6)
    breakdown = {name: torch.nan_to_num(value, nan=1e6, posinf=1e6, neginf=1e6) for name, value in breakdown.items()}
    total_scalar = per_env.mean()
    return total_scalar, per_env, breakdown


__all__ = ["compute_total_loss"]
