"""Configuration contracts for the pure-kinematic joint MPC RTI backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class JointMpcRtiRuntimeCfg:
    horizon_steps: int = 30
    dt: float = 0.02
    sqp_iterations_per_step: int = 1
    max_field_age_steps: int = 2
    dtype: torch.dtype = torch.float32


@dataclass
class JointMpcRtiSolverCfg:
    regularization: float = 1.0e-4
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0)
    line_search_tie_tolerance: float = 1.0e-7
    root_position_trust: float = 0.005
    root_orientation_trust: float = 0.10
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
    foot_contact_offset: float = 0.022
    nominal_joint_pos: tuple[float, ...] = (0.0, 0.8, -1.5) * 4


@dataclass
class JointMpcRtiNominalCfg:
    command_scale: float = 1.0
    step_reference_scale: float = 1.0
    unreachable_step_scale: float = 0.5
    measurement_decay_nodes: int = 6


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

    @property
    def virtual_wall_height(self) -> float:
        return float(self.h_wall)


@dataclass
class JointMpcRtiLossTermsCfg:
    command_linear: float = 1.0
    command_yaw: float = 0.5
    step_xy: float = 1.0
    step_z: float = 0.5
    contact_anchor_xy: float = 100.0
    contact_ground: float = 4.0
    swing_speed_margin: float = 0.002
    terrain_temperature: float = 0.015
    terrain_foot_margin: float = 0.01
    terrain_link_margin: float = 0.015
    terrain_base_margin: float = 0.025
    terrain_touchdown_avoidance: float = 1.0
    posture_root_clearance: float = 0.34
    posture_root_height: float = 1.0
    posture_roll_pitch: float = 1.0
    posture_joint: float = 0.1
    smooth_first: float = 0.02
    smooth_second: float = 1.0


@dataclass
class JointMpcRtiLossCfg:
    command: float = 1.0
    step: float = 1.0
    contact: float = 100.0
    swing_speed: float = 1.0
    terrain: float = 1.0
    posture: float = 1.0
    smooth: float = 1.0

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
class JointMpcRtiCfg:
    runtime: JointMpcRtiRuntimeCfg = field(default_factory=JointMpcRtiRuntimeCfg)
    solver: JointMpcRtiSolverCfg = field(default_factory=JointMpcRtiSolverCfg)
    gait: JointMpcRtiGaitCfg = field(default_factory=JointMpcRtiGaitCfg)
    nominal: JointMpcRtiNominalCfg = field(default_factory=JointMpcRtiNominalCfg)
    terrain: JointMpcRtiTerrainCfg = field(default_factory=JointMpcRtiTerrainCfg)
    loss_terms: JointMpcRtiLossTermsCfg = field(default_factory=JointMpcRtiLossTermsCfg)
    losses: JointMpcRtiLossCfg = field(default_factory=JointMpcRtiLossCfg)


__all__ = [
    "JointMpcRtiCfg",
    "JointMpcRtiGaitCfg",
    "JointMpcRtiLossCfg",
    "JointMpcRtiLossTermsCfg",
    "JointMpcRtiNominalCfg",
    "JointMpcRtiRuntimeCfg",
    "JointMpcRtiSolverCfg",
    "JointMpcRtiTerrainCfg",
]
