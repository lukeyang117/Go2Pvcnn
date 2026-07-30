from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class OfficialCollisionShapeSpec:
    name: str
    leg_name: str | None
    link_type: str
    center_l: tuple[float, float, float]
    quat_wxyz_l: tuple[float, float, float, float]
    shape_type: str
    size_l: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_m: float = 0.0
    height_m: float = 0.0


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
    standstill_fallback_enabled: bool = True
    foot_radius_m: float = 0.022
    knee_radius_m: float = 0.030
    calf_radius_m: float = 0.015
    thigh_radius_m: float = 0.035
    capsule_samples: int = 5
    box_surface_points: int = 6
    cylinder_layers: int = 1
    cylinder_angles: int = 4
    sphere_surface_points: int = 6
    contact_tolerant_collision_shape_names: tuple[str, ...] = ("calf_lower_cylinder", "foot_sphere")
    official_collision_shapes: tuple[OfficialCollisionShapeSpec, ...] = (
        OfficialCollisionShapeSpec(
            "thigh_box",
            None,
            "thigh",
            (0.0, 0.0, -0.1065),
            (0.7071067690849304, 0.0, 0.7071067690849304, 0.0),
            "box",
            size_l=(0.2130, 0.0245, 0.0340),
        ),
        OfficialCollisionShapeSpec(
            "fl_calf_upper_cylinder",
            "FL",
            "calf",
            (0.0080, 0.0, -0.0600),
            (0.9944925904273987, 0.0, -0.10480716824531555, 0.0),
            "cylinder",
            radius_m=0.0120,
            height_m=0.1200,
        ),
        OfficialCollisionShapeSpec(
            "calf_upper_cylinder",
            "!FL",
            "calf",
            (0.0100, 0.0, -0.0600),
            (0.9950041770935059, 0.0, -0.0998334214091301, 0.0),
            "cylinder",
            radius_m=0.0130,
            height_m=0.1200,
        ),
        OfficialCollisionShapeSpec(
            "calf_mid_cylinder",
            None,
            "calf",
            (0.0200, 0.0, -0.1480),
            (0.9996874928474426, 0.0, 0.024997396394610405, 0.0),
            "cylinder",
            radius_m=0.0110,
            height_m=0.0650,
        ),
        OfficialCollisionShapeSpec(
            "calf_lower_cylinder",
            None,
            "calf",
            (0.008013331331312656, 0.0, -0.18745021522045135),
            (0.9650924801826477, 0.0, 0.26190924644470215, 0.0),
            "cylinder",
            radius_m=0.0155,
            height_m=0.0300,
        ),
        OfficialCollisionShapeSpec(
            "foot_sphere",
            None,
            "foot",
            (-0.0020, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            "sphere",
            radius_m=0.0220,
        ),
    )
    obstacle_semantic_ids: ClassVar[tuple[int, ...]] = (1, 2)
