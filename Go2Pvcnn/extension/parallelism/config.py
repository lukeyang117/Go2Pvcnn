from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelismCfg:
    horizon: int = 24
    dt: float = 0.02
    half_cycle: int = 12
    candidate_radius_m: float = 0.24
    candidates_per_leg: int = 50
    hip_lateral_bias_m: float = 0.0955
    root_clearance_m: float = 0.30
    swing_height_m: float = 0.08
    landing_tolerance_m: float = 0.025
    collision_margin_m: float = 0.003
    vx_limit: float = 1.0
    vy_limit: float = 0.5
    vyaw_limit: float = 1.0
    foot_radius_m: float = 0.022
    knee_radius_m: float = 0.030
    calf_radius_m: float = 0.015
    thigh_radius_m: float = 0.035
    capsule_samples: int = 5
