"""Configuration contracts for the pure-kinematic joint MPC RTI backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class JointMpcRtiRuntimeCfg:
    horizon_steps: int = 30
    future_frames: int = 30
    state_nodes: int = 31
    dt: float = 0.02
    sqp_iterations_per_step: int = 1
    max_field_age_steps: int = 0
    dtype: torch.dtype = torch.float32


@dataclass
class JointMpcRtiSolverCfg:
    regularization: float = 3.0
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0)
    line_search_tie_tolerance: float = 1.0e-7
    published_stance_tolerance: float = 0.0005
    published_swing_clearance_buffer: float = 0.0
    root_position_trust: float = 0.01
    root_roll_pitch_trust: float = 0.10
    root_yaw_trust: float = 0.02
    joint_trust: float = 0.25
    active_set_refinements: int = 2
    joint_velocity_limit: float = 30.0
    compile_kernels: bool = False
    use_cuda_graph: bool = False


@dataclass
class JointMpcRtiGaitCfg:
    period_steps: int = 24
    swing_steps: int = 12
    stance_steps: int = 12
    h_swing: float = 0.08
    foot_contact_offset: float = 0.0221
    nominal_joint_pos: tuple[float, ...] = (0.0, 0.8, -1.5) * 4


@dataclass
class JointMpcRtiNominalCfg:
    command_scale: float = 0.45
    step_reference_scale: float = 0.5
    unreachable_step_scale: float = 0.5
    measurement_decay_nodes: int = 6
    terminal_command_fill_scale: float = 0.5
    ik_blend_scale: float = 1.0
    swing_outward_offset_m: float = 0.015
    swing_apex_margin_m: float = 0.08


@dataclass
class JointMpcRtiTouchdownCfg:
    candidate_x_m: tuple[float, ...] = (-0.12, -0.06, 0.0, 0.06, 0.12)
    candidate_y_m: tuple[float, ...] = (-0.12, -0.06, 0.0, 0.06, 0.12)
    command_prediction_scale: float = 0.5
    landing_after_margin_m: float = 0.025
    joint_margin_rad: float = 0.10
    corridor_samples: int = 33
    swing_samples: int = 33
    preview_swing_samples: int = 65
    selector_capsule_samples: int = 5
    latch_phase: int = 6
    w_command: float = 8.0
    w_warm: float = 12.0
    w_slope: float = 1.0
    w_roughness: float = 2.0
    w_edge: float = 0.02


@dataclass
class JointMpcRtiRegionCfg:
    cap_m: float = 0.06
    margin_m: float = 0.005
    min_half_extent_m: float = 0.015
    max_plane_residual_m: float = 0.012


@dataclass
class JointMpcRtiTerrainCfg:
    small_sigma_m: float = 0.04
    large_sigma_m: float = 0.10
    small_gain: float = 1.0
    large_gain: float = 1.0
    h_wall: float = 0.35
    kernel_radius_cells: int = 20
    small_ids: tuple[int, ...] = (1,)
    large_ids: tuple[int, ...] = (2,)
    resolution: float = 0.01
    foot_radius_m: float = 0.022
    knee_radius_m: float = 0.030
    calf_radius_m: float = 0.015
    thigh_radius_m: float = 0.035
    base_radius_m: float = 0.120
    foot_margin_m: float = 0.010
    link_margin_m: float = 0.003
    base_margin_m: float = 0.025
    landing_margin_m: float = 0.010
    slope_max_rad: float = 0.60
    roughness_max_m: float = 0.030
    edge_margin_m: float = 0.020
    roughness_radius_m: float = 0.010
    sweep_subdivisions: int = 24
    capsule_samples: int = 17
    stance_ground_tolerance_m: float = 0.012

    @property
    def virtual_wall_height(self) -> float:
        return float(self.h_wall)


@dataclass
class JointMpcRtiLossTermsCfg:
    command_linear: float = 2.0
    command_yaw: float = 0.5
    command_early_swing: float = 0.0
    command_activity_scale: float = 0.01
    command_hold_multiplier: float = 4000.0
    step_xy: float = 1.0
    step_z: float = 4.0
    contact_anchor_xy: float = 400.0
    contact_future_onset_xy: float = 1.0
    contact_ground: float = 32.0
    swing_speed_margin: float = 0.02
    swing_speed_command_scale: float = 0.4
    swing_speed_early: float = 1.0
    terrain_temperature: float = 0.015
    terrain_foot_margin: float = 0.027
    terrain_link_margin: float = 0.015
    terrain_base_margin: float = 0.025
    terrain_touchdown_avoidance: float = 1.0
    posture_root_clearance: float = 0.34
    posture_root_height: float = 1.0
    posture_roll_pitch: float = 16000.0
    posture_joint: float = 1.2
    smooth_first: float = 3.0
    smooth_second: float = 1.0


@dataclass
class JointMpcRtiLossCfg:
    command: float = 90.0
    step: float = 1.0
    contact: float = 3000.0
    swing_speed: float = 400.0
    terrain: float = 8000.0
    posture: float = 1.0
    smooth: float = 14.25

    def weights(self) -> dict[str, float]:
        names = (
            "command",
            "step",
            "contact",
            "swing_speed",
            "terrain",
            "posture",
            "smooth",
        )
        return {name: float(getattr(self, name)) for name in names}


@dataclass
class JointMpcRtiLqCostCfg:
    velocity_linear: float = 90.0
    velocity_yaw: float = 45.0
    posture_joint: float = 1.2
    hold_velocity_scale: float = 0.20
    hold_roughness_scale: float = 0.03
    root_height: float = 1.0
    root_roll_pitch: float = 16000.0
    root_corridor: float = 0.0
    root_rate: float = 3.0
    swing_position: float = 400.0
    swing_velocity: float = 20.0
    touchdown_xy: float = 80.0
    touchdown_z: float = 320.0
    smooth_first: float = 14.25 * 3.0
    smooth_second: float = 14.25
    warm: float = 3.0
    slack_quadratic: float = 1.0e5
    slack_linear: float = 1.0e3


@dataclass
class JointMpcRtiCfg:
    runtime: JointMpcRtiRuntimeCfg = field(default_factory=JointMpcRtiRuntimeCfg)
    solver: JointMpcRtiSolverCfg = field(default_factory=JointMpcRtiSolverCfg)
    gait: JointMpcRtiGaitCfg = field(default_factory=JointMpcRtiGaitCfg)
    nominal: JointMpcRtiNominalCfg = field(default_factory=JointMpcRtiNominalCfg)
    touchdown: JointMpcRtiTouchdownCfg = field(default_factory=JointMpcRtiTouchdownCfg)
    region: JointMpcRtiRegionCfg = field(default_factory=JointMpcRtiRegionCfg)
    terrain: JointMpcRtiTerrainCfg = field(default_factory=JointMpcRtiTerrainCfg)
    loss_terms: JointMpcRtiLossTermsCfg = field(default_factory=JointMpcRtiLossTermsCfg)
    losses: JointMpcRtiLossCfg = field(default_factory=JointMpcRtiLossCfg)
    lq_cost: JointMpcRtiLqCostCfg = field(default_factory=JointMpcRtiLqCostCfg)


__all__ = [
    "JointMpcRtiCfg",
    "JointMpcRtiGaitCfg",
    "JointMpcRtiLossCfg",
    "JointMpcRtiLqCostCfg",
    "JointMpcRtiLossTermsCfg",
    "JointMpcRtiNominalCfg",
    "JointMpcRtiRegionCfg",
    "JointMpcRtiRuntimeCfg",
    "JointMpcRtiSolverCfg",
    "JointMpcRtiTerrainCfg",
    "JointMpcRtiTouchdownCfg",
]
