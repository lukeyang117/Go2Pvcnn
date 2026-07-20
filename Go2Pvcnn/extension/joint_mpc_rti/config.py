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
    root_position_trust: float = 0.05
    root_orientation_trust: float = 0.10
    joint_trust: float = 0.25
    active_set_refinements: int = 2
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
class JointMpcRtiLossCfg:
    command: float = 1.0
    step: float = 1.0
    contact: float = 1.0
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
    losses: JointMpcRtiLossCfg = field(default_factory=JointMpcRtiLossCfg)


__all__ = [
    "JointMpcRtiCfg",
    "JointMpcRtiGaitCfg",
    "JointMpcRtiLossCfg",
    "JointMpcRtiNominalCfg",
    "JointMpcRtiRuntimeCfg",
    "JointMpcRtiSolverCfg",
]
