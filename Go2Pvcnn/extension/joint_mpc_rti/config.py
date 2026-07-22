"""Configuration contracts for the joint MPC RTI backend."""

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
    barrier_relaxation: float = 1.0e-3
    line_search_alphas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)
    root_xy_trust_scale: float = 0.40
    joint_trust_scale: float = 1.00
    compile_kernels: bool = False
    emit_loss_breakdown: bool = True
    diagonal_state_riccati: bool = False
    coupled_state_riccati: bool = True
    stance_equality_penalty: float = 1.0e4
    stance_dual_step: float = 0.5
    stance_equality_tolerance_m: float = 5.0e-4
    stance_target_step_limit_m: float = 4.5e-4
    stance_ground_gap_limit_m: float = 0.012
    stance_ground_penetration_limit_m: float = 0.001
    root_linear_direction_limit: float = 0.2
    root_vertical_direction_limit: float = 0.4
    root_angular_direction_limit: float = 1.0
    joint_direction_limit: float = 1.0
    constraint_reach_fraction: float = 0.015
    recovery_sdf_reach_fraction: float = 0.25
    swing_clearance_reach_fraction: float = 0.5
    joint_position_safety_margin_rad: float = 0.1
    root_lateral_offset_limit_m: float = 0.06
    root_lateral_velocity_error_limit_mps: float = 0.20
    root_roll_pitch_limit_rad: float = 0.10471975511965977
    root_roll_pitch_rate_limit_rps: float = 0.6
    root_yaw_error_limit_rad: float = 0.17453292519943295
    root_yaw_rate_error_limit_rps: float = 0.8
    use_cuda_graph: bool = False


@dataclass
class JointMpcRtiGaitCfg:
    half_cycle_steps: int = 15
    max_swing_extension_steps: int = 10
    nominal_joint_pos: tuple[float, ...] = (0.0, 0.8, -1.5) * 4
    swing_thigh_angle: float = 1.20
    swing_calf_angle: float = -3.00
    swing_return_exponent: float = 12.0
    max_nominal_joint_velocity: float = 9.0
    liftoff_joint_envelope: float = 0.052
    full_swing_reference_speed: float = 0.1
    stance_safety_lookahead_margin: float = 0.0
    stance_ground_recovery_step_m: float = 0.02
    sdf_target_correction_scale: float = 0.6
    foot_contact_offset: float = 0.022
    nominal_swing_clearance: float = 0.015
    small_semantic_clearance: float = 0.04
    small_semantic_height: float = 0.16
    foot_collision_radius: float = 0.022
    knee_collision_radius: float = 0.040
    calf_collision_radius: float = 0.040
    thigh_collision_radius: float = 0.040
    small_collision_margin_xy: float = 0.010
    small_collision_margin_z: float = 0.060
    swing_clearance_kkt_buffer_m: float = 0.006
    small_touchdown_margin: float = 0.052
    recovery_sdf_exit_buffer_m: float = 0.001
    small_safe_landing_margin: float = 0.035
    small_support_safety_margin: float = 0.035
    stance_ground_far_influence_radius: float = 0.50
    stance_ground_far_temperature: float = 0.005
    small_collision_influence_radius: float = 0.10
    small_foot_over_influence_radius: float = 0.08
    small_collision_temperature: float = 0.02
    small_collision_root_xy_scale: float = 1.0
    small_collision_foot_xy_scale: float = 0.20
    small_collision_link_xy_scale: float = 0.30
    small_collision_vertical_scale: float = 0.10
    collision_restoration_gain: float = 25.0
    collision_restoration_speed_gain: float = 25.0
    collision_restoration_speed_scale: float = 0.01
    small_foot_over_phase_exponent: float = 8.0
    small_safe_landing_phase_exponent: float = 1.0
    small_support_safety_temperature: float = 0.007
    small_support_safety_exponent: float = 1.0
    root_support_height_temperature: float = 0.01
    zero_translation_command_deadband: float = 1.0e-4
    command_touchdown_stride_scale: float = 4.0
    startup_root_leak_limit_m: float = 4.0e-4
    startup_foot_lead_target_m: float = 1.5e-3
    startup_root_release_velocity: float = 0.03
    startup_direction_cosine: float = 0.5


@dataclass
class JointMpcRtiCoreLossCfg:
    command_control_weight: float = 4.0
    root_angular_control_weight: float = 400.0
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
    root_roll_pitch: float = 20.0
    root_vertical_velocity: float = 0.2
    root_roll_pitch_rate: float = 0.2
    joint_nominal_posture: float = 0.25
    joint_position_limit_barrier: float = 0.02
    joint_velocity_limit_barrier: float = 0.02
    stance_xy_lock: float = 1100.0
    stance_equality_violation: float = 1.0e4
    stance_ground_contact: float = 24000.0
    stance_ground_far_gain: float = 4.0
    stance_support_viability: float = 250000.0
    stance_slip_velocity: float = 0.01
    swing_nominal_shape: float = 1700.0
    swing_touchdown_target_multiplier: float = 4.0
    terrain_swing_clearance: float = 4.0
    swing_velocity_smoothness: float = 1.0e-4
    touchdown_velocity: float = 1.0e-3
    touchdown_ground_height: float = 12.0
    touchdown_valid_map: float = 10.0
    touchdown_reach_margin: float = 0.02
    touchdown_foot_separation: float = 0.05
    foot_ground_penetration: float = 60.0
    knee_ground_clearance: float = 2.0
    shank_ground_clearance: float = 2.0
    body_ground_clearance: float = 8.0
    small_object_foot_over: float = 60.0
    small_object_safe_landing: float = 2000.0
    small_object_touchdown_avoidance: float = 150.0
    small_object_foot_clearance: float = 60.0
    small_object_knee_clearance: float = 125.0
    small_object_calf_clearance: float = 125.0
    small_object_thigh_clearance: float = 135.0
    small_object_base_clearance: float = 160.0
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
