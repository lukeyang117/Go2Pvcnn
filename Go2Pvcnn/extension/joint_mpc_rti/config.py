"""Configuration contracts for the joint MPC RTI backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class JointMpcRtiRuntimeCfg:
    horizon_steps: int = 16
    dt: float = 0.02
    sqp_iterations_per_step: int = 1
    max_field_age_steps: int = 2
    dtype: torch.dtype = torch.float32


@dataclass
class JointMpcRtiSolverCfg:
    regularization: float = 1.0e-4
    barrier_relaxation: float = 1.0e-3
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25)
    root_xy_trust_scale: float = 0.05
    joint_trust_scale: float = 0.25


@dataclass
class JointMpcRtiGaitCfg:
    half_cycle_steps: int = 4
    nominal_joint_pos: tuple[float, ...] = (0.0, 0.8, -1.5) * 4
    swing_thigh_angle: float = 0.55
    swing_calf_angle: float = -1.75
    max_nominal_joint_velocity: float = 5.0
    nominal_swing_clearance: float = 0.015
    small_semantic_clearance: float = 0.03


@dataclass
class JointMpcRtiCoreLossCfg:
    command_control_weight: float = 4.0
    joint_velocity_weight: float = 0.4
    joint_posture_weight: float = 0.25
    terminal_joint_posture_weight: float = 0.5
    smoothness_weight: float = 0.01


@dataclass
class JointMpcRtiLossCfg:
    command_linear_velocity: float = 4.0
    command_yaw_rate: float = 2.0
    command_progress: float = 8.0
    command_direction: float = 1.0
    root_support_height: float = 1000.0
    root_roll_pitch: float = 2.0
    root_vertical_velocity: float = 0.2
    root_roll_pitch_rate: float = 0.2
    joint_nominal_posture: float = 0.25
    joint_position_limit_barrier: float = 0.02
    joint_velocity_limit_barrier: float = 0.02
    stance_xy_lock: float = 4.0
    stance_ground_contact: float = 20.0
    stance_slip_velocity: float = 0.01
    swing_nominal_shape: float = 0.2
    terrain_swing_clearance: float = 4.0
    swing_velocity_smoothness: float = 1.0e-4
    touchdown_velocity: float = 1.0e-3
    touchdown_ground_height: float = 12.0
    touchdown_valid_map: float = 10.0
    touchdown_reach_margin: float = 0.02
    touchdown_foot_separation: float = 0.05
    foot_ground_penetration: float = 6.0
    knee_ground_clearance: float = 2.0
    shank_ground_clearance: float = 2.0
    body_ground_clearance: float = 8.0
    small_object_foot_over: float = 10.0
    small_object_touchdown_avoidance: float = 8.0
    small_object_link_clearance: float = 5.0
    large_root_footprint_barrier: float = 30.0
    large_body_collision: float = 12.0
    large_foot_collision: float = 6.0
    large_knee_shank_collision: float = 6.0
    large_terminal_risk: float = 20.0
    control_rate: float = 0.01
    first_control_continuity: float = 0.02
    joint_acceleration: float = 1.0e-5
    root_acceleration: float = 1.0e-4
    terminal_command_velocity: float = 2.0
    terminal_obstacle_safety: float = 10.0
    terminal_posture: float = 0.5
    terminal_contact_viability: float = 0.2


@dataclass
class JointMpcRtiCfg:
    runtime: JointMpcRtiRuntimeCfg = field(default_factory=JointMpcRtiRuntimeCfg)
    solver: JointMpcRtiSolverCfg = field(default_factory=JointMpcRtiSolverCfg)
    gait: JointMpcRtiGaitCfg = field(default_factory=JointMpcRtiGaitCfg)
    core_losses: JointMpcRtiCoreLossCfg = field(default_factory=JointMpcRtiCoreLossCfg)
    losses: JointMpcRtiLossCfg = field(default_factory=JointMpcRtiLossCfg)


__all__ = [
    "JointMpcRtiCfg",
    "JointMpcRtiCoreLossCfg",
    "JointMpcRtiGaitCfg",
    "JointMpcRtiLossCfg",
    "JointMpcRtiRuntimeCfg",
    "JointMpcRtiSolverCfg",
]
