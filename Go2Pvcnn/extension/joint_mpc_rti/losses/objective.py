"""Exact seven-key nonlinear trajectory objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.command import command_residual
from extension.joint_mpc_rti.losses.contact import contact_residual
from extension.joint_mpc_rti.losses.posture import posture_residual
from extension.joint_mpc_rti.losses.smoothness import smooth_residual
from extension.joint_mpc_rti.losses.step import step_residual
from extension.joint_mpc_rti.losses.swing_speed import swing_speed_residual
from extension.joint_mpc_rti.losses.terrain import terrain_residual
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.types import JointMpcTerrainField


LOSS_NAMES = ("command", "step", "contact", "swing_speed", "terrain", "posture", "smooth")


@dataclass(frozen=True)
class LossContext:
    command_body: Tensor
    touchdown_reference_w: Tensor
    schedule: FixedTrotSchedule
    terrain: JointMpcTerrainField
    stance_anchor_w: Tensor
    support_height: Tensor


def trajectory_residuals(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> dict[str, Tensor]:
    return {
        "command": command_residual(state, context.command_body, cfg),
        "step": step_residual(state, context.touchdown_reference_w, context.schedule, cfg),
        "contact": contact_residual(state, context, cfg),
        "swing_speed": swing_speed_residual(state, context.schedule, cfg),
        "terrain": terrain_residual(state, context, cfg),
        "posture": posture_residual(state, context.support_height, cfg),
        "smooth": smooth_residual(state, cfg),
    }


def trajectory_loss_breakdown(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> dict[str, Tensor]:
    residuals = trajectory_residuals(state, context, cfg)
    return {name: 0.5 * residuals[name].square().sum(dim=1) for name in LOSS_NAMES}


def weighted_trajectory_residual(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> Tensor:
    residuals = trajectory_residuals(state, context, cfg)
    weights = cfg.losses.weights()
    return torch.cat(
        tuple(math.sqrt(weights[name]) * residuals[name] for name in LOSS_NAMES),
        dim=1,
    )


def total_trajectory_loss(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> Tensor:
    breakdown = trajectory_loss_breakdown(state, context, cfg)
    weights = cfg.losses.weights()
    total = torch.zeros_like(breakdown[LOSS_NAMES[0]])
    for name in LOSS_NAMES:
        total = total + weights[name] * breakdown[name]
    return total


__all__ = [
    "LOSS_NAMES",
    "LossContext",
    "total_trajectory_loss",
    "trajectory_loss_breakdown",
    "trajectory_residuals",
    "weighted_trajectory_residual",
]
