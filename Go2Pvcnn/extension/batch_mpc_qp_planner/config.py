"""Configuration for the experimental MPC-QP backend."""

from __future__ import annotations

from dataclasses import dataclass, field

from extension.batch_mpc_planner.config import MpcPlannerCfg, MpcRuntimeCfg


@dataclass
class MpcQpRuntimeCfg(MpcRuntimeCfg):
    """QP-specific runtime controls layered on top of the MPC config."""

    qp_iterations: int = 1
    touchdown_keepout_margin_m: float = 0.04
    semantic_repair_radius_m: float = 0.18
    semantic_repair_ring_step_m: float = 0.04
    terrain_step_cap_base_m: float = 0.28
    terrain_step_cap_min_scale: float = 0.55
    terrain_height_variation_threshold_m: float = 0.06
    terrain_step_cap_sample_count: int = 5
    low_small_swing_repair_radius_m: float = 0.10
    low_small_swing_repair_step_m: float = 0.025
    low_small_swing_clearance_m: float = 0.06
    low_small_swing_clearance_max_m: float = 0.16
    low_small_swing_height_lower_step_m: float = 0.20
    low_small_swing_xy_blend: float = 0.65
    low_small_contact_reland_forward_m: float = 0.16
    body_leg_xy_repair_radius_m: float = 0.12
    body_leg_xy_repair_step_m: float = 0.04
    body_leg_xy_repair_gain: float = 1.35
    body_leg_xy_repair_height_margin_m: float = 0.03
    body_leg_xy_repair_passes: int = 2
    body_leg_semantic_clearance_m: float = 0.18
    body_leg_root_lift_margin_m: float = 0.08
    body_leg_root_lift_max_m: float = 0.20
    low_small_crossing_root_lift_m: float = 0.08
    continuous_trajectory_enabled: bool = True
    continuous_bezier_sample_count: int = 0
    continuous_footprint_radius_m: float = 0.04
    continuous_foothold_step_m: float = 0.025
    continuous_foothold_variation_target_m: float = 0.03
    continuous_foothold_probe_m: float = 0.04
    continuous_fk_readback_gain: float = 0.75
    continuous_fk_readback_max_step_m: float = 0.06
    continuous_fk_root_z_gain: float = 0.85
    continuous_fk_root_z_max_step_m: float = 0.06
    continuous_fk_root_z_error_threshold_m: float = 0.08
    continuous_fk_root_xy_gain: float = 0.5
    continuous_fk_root_xy_max_step_m: float = 0.10
    continuous_low_small_final_readback_passes: int = 2
    continuous_fk_endpoint_gain: float = 0.5
    continuous_fk_endpoint_max_step_m: float = 0.20
    continuous_joint_limit_readback_gain: float = 0.75
    continuous_joint_limit_readback_max_step_m: float = 0.10
    continuous_joint_limit_endpoint_gain: float = 0.35
    continuous_joint_limit_endpoint_max_step_m: float = 0.10
    continuous_start_tangent_scale: float = 0.20
    continuous_reachability_step_m: float = 0.45
    continuous_terrain_clearance_m: float = 0.018
    continuous_terrain_clearance_step_m: float = 0.09
    continuous_plane_root_height_min_m: float = 0.30
    continuous_plane_root_height_max_m: float = 0.36
    continuous_low_small_progress_lookahead_m: float = 0.48
    continuous_low_small_progress_sample_count: int = 17
    continuous_low_small_progress_lane_half_width_m: float = 0.14
    continuous_low_small_progress_margin_m: float = 0.06
    continuous_low_small_progress_step_m: float = 0.16
    continuous_low_small_foot_over_lateral_step_m: float = 0.04
    continuous_low_small_foot_over_endpoint_margin_m: float = 0.08
    continuous_low_small_foot_over_lift_m: float = 0.06
    continuous_low_small_crossing_endpoint_step_m: float = 0.10
    continuous_low_small_crossing_arc_margin_m: float = 0.04
    continuous_low_small_crossing_arc_lift_step_m: float = 0.10
    continuous_low_small_crossing_arc_lateral_step_m: float = 0.08
    continuous_low_small_crossing_arc_lane_margin_m: float = 0.08
    continuous_low_small_crossing_arc_target_lane_m: float = 0.03


@dataclass
class MpcQpPlannerCfg(MpcPlannerCfg):
    """MPC-QP config.

    The inherited MPC fields preserve the current scanner/runtime/loss
    contracts. ``runtime.qp_iterations`` is consumed only by the QP backend.
    """

    runtime: MpcQpRuntimeCfg = field(default_factory=MpcQpRuntimeCfg)


def planner_cfg_from_task_cfg(task_cfg) -> MpcQpPlannerCfg:
    """Return the QP planner cfg from a task cfg, preserving explicit fields."""

    existing = getattr(task_cfg, "mpc_qp_planner_cfg", None)
    if existing is None:
        return MpcQpPlannerCfg()
    if not isinstance(existing, MpcQpPlannerCfg):
        raise TypeError(f"mpc_qp_planner_cfg must be MpcQpPlannerCfg, got {type(existing).__name__}")
    return existing


def validate_mpc_qp_config(cfg: MpcQpPlannerCfg) -> None:
    if int(cfg.runtime.qp_iterations) <= 0:
        raise ValueError("runtime.qp_iterations must be positive")


__all__ = ["MpcQpPlannerCfg", "MpcQpRuntimeCfg", "planner_cfg_from_task_cfg", "validate_mpc_qp_config"]
