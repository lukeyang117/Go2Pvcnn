"""Fixed-contract native torch planner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

FIXED_HORIZON_S = 0.7
FIXED_DT = 0.02
FIXED_HORIZON_STEPS = 35
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
    default_terrain_local_resolution: float = 0.01
    cost_weights: dict[str, float] = field(default_factory=_default_cost_weights)


def validate_config(cfg: TogetherPlannerConfig) -> None:
    if not isinstance(cfg, TogetherPlannerConfig):
        raise TypeError("cfg must be a TogetherPlannerConfig")
    if abs(float(cfg.horizon_s) - FIXED_HORIZON_S) > 1e-12:
        raise ValueError("horizon_s must equal the fixed 0.7s contract")
    if abs(float(cfg.dt) - FIXED_DT) > 1e-12:
        raise ValueError("dt must equal the fixed 0.02s contract")
    if int(cfg.horizon_steps) != FIXED_HORIZON_STEPS:
        raise ValueError("horizon_steps must equal the fixed 35-frame contract")
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
