"""Fixed-contract native torch planner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

FIXED_HORIZON_S = 1.0
FIXED_DT = 0.02
FIXED_HORIZON_STEPS = 50
FIXED_EVENT_CAP = 2


def _default_cost_weights() -> dict[str, float]:
    return {
        "J_td": 3.0,
        "J_swing": 2.0,
        "J_ik": 4.0,
        "J_base": 1.0,
        "J_vel": 0.5,
    }


@dataclass(frozen=True)
class TogetherPlannerConfig:
    horizon_s: float = FIXED_HORIZON_S
    dt: float = FIXED_DT
    horizon_steps: int = FIXED_HORIZON_STEPS
    event_cap: int = FIXED_EVENT_CAP
    step_freq: float = 2.0
    duty_factor: float = 0.55
    leg_phase_offsets: tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0)
    idle_command_eps: float = 1e-4
    seed_count: int = 6
    elite_count: int = 2
    cem_iters: int = 2
    swing_height: float = 0.08
    touchdown_xy_bias: float = 0.10
    touchdown_clearance_margin: float = 0.02
    touchdown_ground_gap_tolerance_m: float = 0.02
    swing_clearance_margin: float = 0.08
    base_min_height: float = 0.28
    feasible_joint_violation_max: float = 1e-4
    feasible_workspace_margin_min: float = -0.02
    safe_workspace_margin_min: float = -0.08
    support_search_radius: float = 0.04
    support_search_step: float = 0.02
    support_walkable_slope: float = 0.6
    slope_sample_step: float = 0.02
    hip_height: float = 0.30
    touchdown_xy_bias_scale: float = 0.04
    touchdown_z_bias_scale: float = 0.02
    swing_parabola_multiplier: float = 4.0
    rehome_roll_pitch_limit: float = 0.35
    default_terrain_local_resolution: float = 0.01
    semantic_small_id: int = 1
    semantic_large_id: int = 2
    candidate_count: int = 5
    semantic_lateral_offset_m: float = 0.45
    semantic_foothold_lateral_scale: float = 0.25
    semantic_lateral_bias_weight: float = 0.05
    semantic_collision_weight: float = 20.0
    semantic_large_collision_weight: float = 80.0
    small_crossable_height_max: float = 0.28
    small_foot_clearance: float = 0.06
    small_body_clearance: float = 0.04
    touchdown_small_boundary_penalty_margin: float = 0.03
    touchdown_small_boundary_invalidation_margin: float = 0.01
    touchdown_small_boundary_penalty_weight: float = 12.0
    front_pair_consistency_penalty_margin: float = 0.06
    front_pair_consistency_invalidation_margin: float = 0.10
    rear_pair_follow_penalty_margin: float = 0.06
    rear_pair_follow_invalidation_margin: float = 0.10
    body_posture_penalty_margin: float = 0.10
    body_posture_invalidation_margin: float = 0.20
    front_pair_consistency_weight: float = 18.0
    rear_pair_follow_weight: float = 18.0
    body_posture_weight: float = 14.0
    state_bypass_center_penalty_weight: float = 12.0
    candidate_path_foot_penalty_margin: float = 0.03
    candidate_path_foot_invalidation_margin: float = 0.0
    candidate_path_leg_penalty_margin: float = 0.03
    candidate_path_leg_invalidation_margin: float = 0.0
    candidate_path_clearance_weight: float = 12.0
    large_body_clearance: float = 0.08
    max_root_lift_for_small: float = 0.10
    body_footprint_forward_m: float = 0.28
    body_footprint_lateral_m: float = 0.14
    body_footprint_sample_count: int = 9
    body_underside_offset_m: float = 0.08
    swing_height_query_count: int = 9
    swing_height_clearance_margin: float = 0.04
    body_collision_sample_count: int = 17
    body_collision_soft_margin: float = 0.05
    body_collision_hard_penetration_m: float = 0.01
    body_collision_weight: float = 10.0
    leg_collision_axis_sample_count: int = 5
    leg_collision_radius_m: float = 0.01
    leg_collision_soft_margin: float = 0.03
    leg_collision_hard_penetration_m: float = 0.01
    leg_collision_weight: float = 8.0
    command_retention_weight: float = 3.2
    semantic_reference_radius: float = 0.06
    semantic_reference_sample_count: int = 9
    cost_weights: dict[str, float] = field(default_factory=_default_cost_weights)


def validate_config(cfg: TogetherPlannerConfig) -> None:
    if not isinstance(cfg, TogetherPlannerConfig):
        raise TypeError("cfg must be a TogetherPlannerConfig")
    if abs(float(cfg.horizon_s) - FIXED_HORIZON_S) > 1e-12:
        raise ValueError("horizon_s must equal the fixed 1.0s contract")
    if abs(float(cfg.dt) - FIXED_DT) > 1e-12:
        raise ValueError("dt must equal the fixed 0.02s contract")
    if int(cfg.horizon_steps) != FIXED_HORIZON_STEPS:
        raise ValueError("horizon_steps must equal the fixed 50-step contract")
    if int(cfg.event_cap) != FIXED_EVENT_CAP:
        raise ValueError("event_cap must equal the fixed 2-event contract")


__all__ = [
    "FIXED_DT",
    "FIXED_EVENT_CAP",
    "FIXED_HORIZON_S",
    "FIXED_HORIZON_STEPS",
    "TogetherPlannerConfig",
    "validate_config",
]
