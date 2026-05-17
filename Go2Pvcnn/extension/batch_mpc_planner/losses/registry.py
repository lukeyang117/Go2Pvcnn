"""Loss registry for batch MPC."""

from __future__ import annotations

import torch
from torch import Tensor

from ..config import MpcPlannerCfg
from ..types import MpcPlannerTerrain, MpcRobotState
from ..variables import DecodedMpcTrajectory
from .contact import contact_binary_loss, contact_transition_loss, support_stability_loss
from .gait_coupling import (
    diagonal_pair_loss,
    phase_prior_loss,
    root_foot_center_loss,
    root_height_loss,
    support_plane_roll_pitch_loss,
    swing_center_urgency_order_loss,
    swing_direction_loss,
    swing_window_loss,
)
from .kinematics import evaluate_kinematics_for_loss, ik_fk_residual_loss, joint_limit_loss_from_root_foot
from .smoothness import foot_smoothness_loss, root_smoothness_loss
from .terrain_clearance import (
    body_heightfield_collision_loss,
    finite_horizon_touchdown_phase,
    high_obstacle_avoidance_loss,
    knee_shank_heightfield_collision_loss,
    low_small_crossing_progress_loss,
    obstacle_risk_scales,
    sample_time,
    semantic_contact_avoidance_loss,
    semantic_obstacle_loss,
    stance_ground_loss,
    stance_semantic_obstacle_loss,
    swing_clearance_terrain_loss,
    touchdown_semantic_loss,
    touchdown_surface_loss,
)
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


def compute_total_loss(
    decoded: DecodedMpcTrajectory,
    nominal: dict[str, Tensor],
    state: MpcRobotState,
    command: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcPlannerCfg,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Return scalar loss, per-env loss, and term breakdown."""
    losses = cfg.losses
    runtime = cfg.runtime
    breakdown: dict[str, Tensor] = {}
    per_env = torch.zeros(decoded.root_pos.shape[0], dtype=decoded.root_pos.dtype, device=decoded.root_pos.device)

    risk_scales = obstacle_risk_scales(
        terrain,
        decoded.root_pos,
        decoded.root_rpy,
        command,
        small_ids=losses.touchdown_semantic.small_ids,
        large_ids=losses.touchdown_semantic.large_ids,
        high_small_relative_height_m=losses.obstacle_risk.high_small_relative_height_m,
        linear_corridor_width_m=losses.obstacle_risk.linear_corridor_width_m,
        linear_forward_distance_m=losses.obstacle_risk.linear_forward_distance_m,
        yaw_swept_radius_m=losses.obstacle_risk.yaw_swept_radius_m,
        linear_scale_when_blocked=losses.obstacle_risk.linear_scale_when_blocked,
        yaw_scale_when_blocked=losses.obstacle_risk.yaw_scale_when_blocked,
        linear_speed_eps=losses.obstacle_risk.linear_speed_eps,
        yaw_speed_eps=losses.obstacle_risk.yaw_speed_eps,
    )
    linear_scale = risk_scales.linear_scale if losses.obstacle_risk.enabled else None
    yaw_scale = risk_scales.yaw_scale if losses.obstacle_risk.enabled else None
    breakdown["obstacle_risk_linear_scale"] = risk_scales.linear_scale
    breakdown["obstacle_risk_yaw_scale"] = risk_scales.yaw_scale
    breakdown["obstacle_risk_linear_trigger_count"] = risk_scales.linear_trigger_count.to(dtype=per_env.dtype)
    breakdown["obstacle_risk_yaw_trigger_count"] = risk_scales.yaw_trigger_count.to(dtype=per_env.dtype)
    breakdown["obstacle_risk_trigger_horizon_index"] = risk_scales.trigger_horizon_index.to(dtype=per_env.dtype)
    breakdown["obstacle_risk_trigger_semantic_class"] = risk_scales.trigger_semantic_class.to(dtype=per_env.dtype)

    track = command_tracking_loss(
        decoded.root_pos,
        decoded.root_rpy,
        command,
        runtime.dt,
        vel_weight=losses.tracking.vel_weight,
        yaw_weight=losses.tracking.yaw_weight,
        linear_scale=linear_scale,
        yaw_scale=yaw_scale,
    )
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
        contact_threshold=runtime.contact_threshold,
    )
    contact_loss = (
        losses.contact_regularization.binary_weight * cbin
        + losses.contact_regularization.transition_weight * ctran
        + support
    )
    per_env = per_env + _weighted(
        losses.contact_regularization.enabled,
        losses.contact_regularization.weight,
        contact_loss,
        breakdown,
        "contact_regularization",
    )

    window = swing_window_loss(decoded.swing_width, nominal, runtime)
    window = window + losses.swing_window.phase_prior_weight * phase_prior_loss(decoded.swing_center, decoded.swing_width, nominal)
    per_env = per_env + _weighted(losses.swing_window.enabled, losses.swing_window.weight, window, breakdown, "swing_window")

    diagonal = diagonal_pair_loss(decoded.swing_center, decoded.swing_width)
    per_env = per_env + _weighted(losses.diagonal_pair.enabled, losses.diagonal_pair.weight, diagonal, breakdown, "diagonal_pair")

    urgency = swing_center_urgency_order_loss(
        decoded.swing_center,
        decoded.swing_width,
        state,
        command,
        runtime,
        terrain=terrain,
        nominal=nominal,
    )
    per_env = per_env + _weighted(
        losses.swing_center_urgency.enabled,
        losses.swing_center_urgency.weight,
        urgency,
        breakdown,
        "swing_center_urgency",
    )

    stance = stance_ground_loss(
        terrain,
        decoded.foot_pos,
        decoded.contact_prob,
        min_contact_prob=runtime.contact_threshold,
    )
    per_env = per_env + _weighted(losses.stance_ground.enabled, losses.stance_ground.weight, stance, breakdown, "stance_ground")

    swing_clear = swing_clearance_terrain_loss(
        terrain,
        decoded.foot_pos,
        decoded.swing_prob,
        min_clearance_m=losses.swing_clearance_terrain.min_clearance_m,
        worst_deficit_weight=losses.swing_clearance_terrain.worst_deficit_weight,
        min_swing_prob=1.0 - float(runtime.contact_threshold),
        hard_active_weight=True,
        boundary_min_swing_prob=losses.swing_clearance_terrain.boundary_min_swing_prob,
        boundary_weight=losses.swing_clearance_terrain.boundary_weight,
    )
    per_env = per_env + _weighted(
        losses.swing_clearance_terrain.enabled,
        losses.swing_clearance_terrain.weight,
        swing_clear,
        breakdown,
        "swing_clearance_terrain",
    )

    touchdown_phase = finite_horizon_touchdown_phase(decoded.swing_center, decoded.swing_width)
    touchdown_w = sample_time(decoded.foot_pos, touchdown_phase, cyclic=False)
    td_surface = touchdown_surface_loss(
        terrain,
        touchdown_w,
        slope_sample_step=losses.touchdown_surface.slope_sample_step_m,
        support_search_radius=losses.touchdown_surface.support_search_radius_m,
        support_search_step=losses.touchdown_surface.support_search_step_m,
        max_slope=losses.touchdown_surface.max_slope,
        max_support_slope=losses.touchdown_surface.max_support_slope,
        support_height_tolerance=losses.touchdown_surface.support_height_tolerance_m,
        ground_weight=losses.touchdown_surface.ground_weight,
        slope_weight=losses.touchdown_surface.slope_weight,
        support_distance_weight=losses.touchdown_surface.support_distance_weight,
        support_height_weight=losses.touchdown_surface.support_height_weight,
        support_slope_weight=losses.touchdown_surface.support_slope_weight,
        invalid_support_weight=losses.touchdown_surface.invalid_support_weight,
    )
    per_env = per_env + _weighted(
        losses.touchdown_surface.enabled,
        losses.touchdown_surface.weight,
        td_surface,
        breakdown,
        "touchdown_surface",
    )

    td_sem = touchdown_semantic_loss(
        terrain,
        touchdown_w[..., :2],
        touchdown_w[..., 2],
        small_weight=losses.touchdown_semantic.small_weight,
        large_weight=losses.touchdown_semantic.large_weight,
    ).to(dtype=per_env.dtype, device=per_env.device)
    per_env = per_env + _weighted(
        losses.touchdown_semantic.enabled,
        losses.touchdown_semantic.weight,
        td_sem,
        breakdown,
        "touchdown_semantic",
    )

    stance_sem = stance_semantic_obstacle_loss(
        terrain,
        decoded.foot_pos,
        decoded.contact_prob,
        ground_ids=losses.stance_semantic.ground_ids,
        small_ids=losses.stance_semantic.small_ids,
        large_ids=losses.stance_semantic.large_ids,
        small_weight=losses.stance_semantic.small_weight,
        large_weight=losses.stance_semantic.large_weight,
        min_contact_prob=runtime.contact_threshold,
    )
    per_env = per_env + _weighted(
        losses.stance_semantic.enabled,
        losses.stance_semantic.weight,
        stance_sem,
        breakdown,
        "stance_semantic",
    )

    semantic_contact = semantic_contact_avoidance_loss(
        terrain,
        decoded.foot_pos,
        decoded.contact_prob,
        ground_ids=losses.semantic_contact_avoid.ground_ids,
        small_ids=losses.semantic_contact_avoid.small_ids,
        large_ids=losses.semantic_contact_avoid.large_ids,
        small_weight=losses.semantic_contact_avoid.small_weight,
        large_weight=losses.semantic_contact_avoid.large_weight,
        activation_margin=losses.semantic_contact_avoid.activation_margin,
        worst_contact_weight=losses.semantic_contact_avoid.worst_contact_weight,
        soft_margin_m=losses.semantic_contact_avoid.soft_margin_m,
        soft_field_weight=losses.semantic_contact_avoid.soft_field_weight,
        soft_worst_field_weight=losses.semantic_contact_avoid.soft_worst_field_weight,
    )
    per_env = per_env + _weighted(
        losses.semantic_contact_avoid.enabled,
        losses.semantic_contact_avoid.weight,
        semantic_contact,
        breakdown,
        "semantic_contact_avoid",
    )

    obstacle = semantic_obstacle_loss(
        terrain,
        decoded.root_pos,
        decoded.root_rpy,
        decoded.foot_pos,
        decoded.contact_prob,
        decoded.swing_prob,
        small_weight=losses.semantic_obstacle.small_weight,
        large_weight=losses.semantic_obstacle.large_weight,
        body_weight=losses.semantic_obstacle.body_weight,
        foot_weight=losses.semantic_obstacle.foot_weight,
        body_stencil_radius_m=losses.semantic_obstacle.body_stencil_radius_m,
        soft_margin_m=losses.semantic_obstacle.soft_margin_m,
        body_soft_field_weight=losses.semantic_obstacle.body_soft_field_weight,
        body_soft_worst_field_weight=losses.semantic_obstacle.body_soft_worst_field_weight,
        foot_soft_field_weight=losses.semantic_obstacle.foot_soft_field_weight,
        foot_soft_worst_field_weight=losses.semantic_obstacle.foot_soft_worst_field_weight,
        high_small_relative_height_m=losses.semantic_obstacle.high_small_relative_height_m,
    )
    per_env = per_env + _weighted(
        losses.semantic_obstacle.enabled,
        losses.semantic_obstacle.weight,
        obstacle,
        breakdown,
        "semantic_obstacle",
    )

    low_small_crossing = low_small_crossing_progress_loss(
        terrain,
        decoded.root_pos,
        decoded.root_rpy,
        command,
        small_ids=losses.touchdown_semantic.small_ids,
        high_small_relative_height_m=losses.low_small_crossing.high_small_relative_height_m,
        corridor_width_m=losses.low_small_crossing.corridor_width_m,
        forward_distance_m=losses.low_small_crossing.forward_distance_m,
        pass_margin_m=losses.low_small_crossing.pass_margin_m,
        obstacle_depth_m=losses.low_small_crossing.obstacle_depth_m,
        linear_speed_eps=losses.low_small_crossing.linear_speed_eps,
    )
    per_env = per_env + _weighted(
        losses.low_small_crossing.enabled,
        losses.low_small_crossing.weight,
        low_small_crossing,
        breakdown,
        "low_small_crossing",
    )

    high_obstacle_avoidance = high_obstacle_avoidance_loss(
        terrain,
        decoded.root_pos,
        decoded.root_rpy,
        command,
        small_ids=losses.touchdown_semantic.small_ids,
        large_ids=losses.touchdown_semantic.large_ids,
        high_small_relative_height_m=losses.high_obstacle_avoidance.high_small_relative_height_m,
        corridor_width_m=losses.high_obstacle_avoidance.corridor_width_m,
        forward_distance_m=losses.high_obstacle_avoidance.forward_distance_m,
        lateral_clearance_m=losses.high_obstacle_avoidance.lateral_clearance_m,
        longitudinal_influence_m=losses.high_obstacle_avoidance.longitudinal_influence_m,
        linear_speed_eps=losses.high_obstacle_avoidance.linear_speed_eps,
    )
    per_env = per_env + _weighted(
        losses.high_obstacle_avoidance.enabled,
        losses.high_obstacle_avoidance.weight,
        high_obstacle_avoidance,
        breakdown,
        "high_obstacle_avoidance",
    )

    body_collision = body_heightfield_collision_loss(
        terrain,
        decoded.root_pos,
        decoded.root_rpy,
        bottom_offset_z=losses.body_collision.bottom_offset_z_m,
        margin_m=losses.body_collision.margin_m,
        stencil_xy=losses.body_collision.stencil_xy_m,
    )
    per_env = per_env + _weighted(
        losses.body_collision.enabled,
        losses.body_collision.weight,
        body_collision,
        breakdown,
        "body_collision",
    )

    loss_kinematics = evaluate_kinematics_for_loss(
        decoded.root_pos,
        decoded.root_rpy,
        decoded.foot_pos,
        clamp_to_limits=True,
        shank_sample_count=losses.leg_collision.shank_sample_count,
    )
    leg_collision = knee_shank_heightfield_collision_loss(
        terrain,
        loss_kinematics.leg_points.knee_pos_world,
        loss_kinematics.leg_points.shank_sample_world,
        knee_margin_m=losses.leg_collision.knee_margin_m,
        shank_margin_m=losses.leg_collision.shank_margin_m,
        worst_deficit_weight=losses.leg_collision.worst_deficit_weight,
    )
    per_env = per_env + _weighted(
        losses.leg_collision.enabled,
        losses.leg_collision.weight,
        leg_collision,
        breakdown,
        "leg_collision",
    )

    swing_dir = swing_direction_loss(
        decoded.root_pos,
        decoded.root_rpy,
        decoded.foot_pos,
        decoded.swing_center,
        decoded.swing_width,
        command,
        runtime,
    )
    per_env = per_env + _weighted(losses.swing_direction.enabled, losses.swing_direction.weight, swing_dir, breakdown, "swing_direction")

    progress = progress_direction_loss(decoded.root_pos, decoded.root_rpy, command, losses.progress.min_progress_m)
    per_env = per_env + _weighted(losses.progress.enabled, losses.progress.weight, progress, breakdown, "progress")

    kin = joint_limit_loss_from_root_foot(
        decoded.root_pos,
        decoded.root_rpy,
        decoded.foot_pos,
        joint_limit_margin_rad=losses.kinematics.joint_limit_margin_rad,
    )
    per_env = per_env + _weighted(losses.kinematics.enabled, losses.kinematics.weight, kin, breakdown, "ik_joint_limit")

    ik_fk = ik_fk_residual_loss(
        decoded.root_pos,
        decoded.root_rpy,
        decoded.foot_pos,
        decoded.contact_prob,
        contact_weight=losses.ik_fk_residual.contact_weight,
    )
    per_env = per_env + _weighted(
        losses.ik_fk_residual.enabled,
        losses.ik_fk_residual.weight,
        ik_fk,
        breakdown,
        "ik_fk_residual",
    )

    root_center = root_foot_center_loss(decoded.root_pos, decoded.foot_pos)
    per_env = per_env + _weighted(
        losses.root_foot_center.enabled,
        losses.root_foot_center.weight,
        root_center,
        breakdown,
        "root_foot_center",
    )

    root_h = root_height_loss(decoded.root_pos, nominal)
    per_env = per_env + _weighted(
        losses.root_height.enabled,
        losses.root_height.weight,
        root_h,
        breakdown,
        "root_height",
    )

    plane = support_plane_roll_pitch_loss(
        decoded.root_rpy,
        decoded.foot_pos,
        decoded.contact_prob,
        swing_weight=losses.support_plane_rp.swing_weight,
    )
    per_env = per_env + _weighted(
        losses.support_plane_rp.enabled,
        losses.support_plane_rp.weight,
        plane,
        breakdown,
        "support_plane_rp",
    )

    per_env = torch.nan_to_num(per_env, nan=1e6, posinf=1e6, neginf=1e6)
    breakdown = {name: torch.nan_to_num(value, nan=1e6, posinf=1e6, neginf=1e6) for name, value in breakdown.items()}
    total_scalar = per_env.mean()
    return total_scalar, per_env, breakdown


__all__ = ["compute_total_loss"]
