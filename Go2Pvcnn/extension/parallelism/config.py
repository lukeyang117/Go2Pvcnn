from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class EllipsoidSpec:
    name: str
    link_type: str
    center_l: tuple[float, float, float]
    radii_l: tuple[float, float, float]
    probe_offset_l: tuple[float, float]


@dataclass(frozen=True)
class ParallelismCfg:
    horizon: int = 24
    dt: float = 0.02
    half_cycle: int = 12
    candidate_radius_m: float = 0.24
    candidates_per_leg: int = 50
    hip_lateral_bias_m: float = 0.0955
    foothold_step_gain: float = 1.5
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
    collision_probe_count: int = 5
    contact_tolerant_collision_names: tuple[str, ...] = ("calf_ankle_cap", "foot_pad")
    collision_ellipsoids: tuple[EllipsoidSpec, ...] = (
        EllipsoidSpec("thigh_body_inner", "thigh", (0.000, 0.035, -0.005), (0.055, 0.050, 0.035), (0.045, 0.045)),
        EllipsoidSpec("thigh_body_mid", "thigh", (0.000, 0.085, -0.005), (0.060, 0.055, 0.032), (0.050, 0.050)),
        EllipsoidSpec("thigh_body_outer", "thigh", (0.000, 0.135, -0.006), (0.060, 0.050, 0.032), (0.050, 0.045)),
        EllipsoidSpec("thigh_outer_cap", "thigh", (0.000, 0.180, -0.010), (0.050, 0.040, 0.038), (0.040, 0.035)),
        EllipsoidSpec("calf_knee_cap", "calf", (0.006, 0.000, -0.025), (0.030, 0.020, 0.032), (0.024, 0.018)),
        EllipsoidSpec("calf_upper_bar", "calf", (0.010, 0.000, -0.070), (0.026, 0.016, 0.045), (0.022, 0.014)),
        EllipsoidSpec("calf_mid_bar", "calf", (0.016, 0.000, -0.115), (0.024, 0.015, 0.045), (0.020, 0.013)),
        EllipsoidSpec("calf_lower_bar", "calf", (0.018, 0.000, -0.158), (0.023, 0.014, 0.038), (0.019, 0.012)),
        EllipsoidSpec("calf_ankle_cap", "calf", (0.008, 0.000, -0.195), (0.028, 0.018, 0.026), (0.022, 0.015)),
        EllipsoidSpec("foot_pad", "foot", (-0.002, 0.000, 0.030), (0.026, 0.024, 0.022), (0.020, 0.018)),
    )
    obstacle_semantic_ids: ClassVar[tuple[int, ...]] = (1, 2)
