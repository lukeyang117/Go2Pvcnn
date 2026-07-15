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


@dataclass
class JointMpcRtiGaitCfg:
    half_cycle_steps: int = 4
    nominal_joint_pos: tuple[float, ...] = (0.0, 0.8, -1.5) * 4
    swing_thigh_angle: float = 0.55
    swing_calf_angle: float = -1.75
    max_nominal_joint_velocity: float = 5.0


@dataclass
class JointMpcRtiCoreLossCfg:
    command_control_weight: float = 4.0
    joint_velocity_weight: float = 0.4
    joint_posture_weight: float = 0.25
    terminal_joint_posture_weight: float = 0.5
    smoothness_weight: float = 0.01


@dataclass
class JointMpcRtiCfg:
    runtime: JointMpcRtiRuntimeCfg = field(default_factory=JointMpcRtiRuntimeCfg)
    solver: JointMpcRtiSolverCfg = field(default_factory=JointMpcRtiSolverCfg)
    gait: JointMpcRtiGaitCfg = field(default_factory=JointMpcRtiGaitCfg)
    core_losses: JointMpcRtiCoreLossCfg = field(default_factory=JointMpcRtiCoreLossCfg)


__all__ = [
    "JointMpcRtiCfg",
    "JointMpcRtiCoreLossCfg",
    "JointMpcRtiGaitCfg",
    "JointMpcRtiRuntimeCfg",
    "JointMpcRtiSolverCfg",
]
