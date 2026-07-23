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
from extension.joint_mpc_rti.types import JointMpcPerceptiveField, JointMpcTerrainField


LOSS_NAMES = ("command", "step", "contact", "swing_speed", "terrain", "posture", "smooth")


@dataclass(frozen=True)
class LossContext:
    command_body: Tensor
    touchdown_reference_w: Tensor
    schedule: FixedTrotSchedule
    terrain: JointMpcTerrainField
    stance_anchor_w: Tensor
    support_height: Tensor
    perceptive_field: JointMpcPerceptiveField | None = None


def trajectory_residuals(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> dict[str, Tensor]:
    return {
        "command": command_residual(state, context.command_body, context.schedule, cfg),
        "step": step_residual(state, context.touchdown_reference_w, context.schedule, cfg),
        "contact": contact_residual(state, context, cfg),
        "swing_speed": swing_speed_residual(state, context.command_body, context.schedule, cfg),
        "terrain": terrain_residual(state, context, cfg),
        "posture": posture_residual(state, context.support_height, cfg),
        "smooth": smooth_residual(state, cfg),
    }


def _loss_breakdown_from_residuals(residuals: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: 0.5 * residuals[name].square().sum(dim=1) for name in LOSS_NAMES}


def _node_loss_breakdown_from_residuals(
    state: Tensor, residuals: dict[str, Tensor]
) -> dict[str, Tensor]:
    trajectory = torch.as_tensor(state)
    batch, nodes, state_dim = trajectory.shape

    step = 0.5 * residuals["step"].reshape(batch, nodes, 4, 3).square().sum(dim=(2, 3))

    terrain = trajectory.new_zeros(batch, nodes)
    cursor = 0
    for width in (4, 4, 12, 12, 9, 4):
        count = nodes * width
        block = residuals["terrain"][:, cursor : cursor + count].reshape(batch, nodes, width)
        terrain = terrain + 0.5 * block.square().sum(dim=2)
        cursor += count

    smooth = trajectory.new_zeros(batch, nodes)
    first_count = (nodes - 1) * state_dim
    first = residuals["smooth"][:, :first_count].reshape(batch, nodes - 1, state_dim)
    second = residuals["smooth"][:, first_count:].reshape(batch, nodes - 2, state_dim)
    smooth[:, 1:] = smooth[:, 1:] + 0.5 * first.square().sum(dim=2)
    smooth[:, 1:-1] = smooth[:, 1:-1] + 0.5 * second.square().sum(dim=2)
    return {"step": step, "terrain": terrain, "smooth": smooth}


def trajectory_loss_breakdown(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> dict[str, Tensor]:
    return _loss_breakdown_from_residuals(trajectory_residuals(state, context, cfg))


def trajectory_node_loss_breakdown(
    state: Tensor, context: LossContext, cfg: JointMpcRtiCfg
) -> dict[str, Tensor]:
    residuals = trajectory_residuals(state, context, cfg)
    return _node_loss_breakdown_from_residuals(state, residuals)


def trajectory_loss_diagnostics(
    state: Tensor, context: LossContext, cfg: JointMpcRtiCfg
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    residuals = trajectory_residuals(state, context, cfg)
    return (
        _loss_breakdown_from_residuals(residuals),
        _node_loss_breakdown_from_residuals(state, residuals),
    )


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
    "trajectory_loss_diagnostics",
    "trajectory_node_loss_breakdown",
    "trajectory_residuals",
    "weighted_trajectory_residual",
]
