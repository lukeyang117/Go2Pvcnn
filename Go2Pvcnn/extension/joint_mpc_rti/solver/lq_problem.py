"""Eight-family direct-state LQ cost and block-banded GGN assembly."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import LossContext
from extension.joint_mpc_rti.model.go2_kinematics import (
    complete_foot_jacobian,
    go2_fk,
)
from extension.joint_mpc_rti.model.nominal import NominalTrajectory


RESIDUAL_FAMILIES = (
    "velocity",
    "posture",
    "root",
    "swing",
    "touchdown",
    "smooth",
    "warm",
    "slack",
)


@dataclass(frozen=True)
class _LocalTerm:
    residual: Tensor
    jacobians: tuple[Tensor, ...]


@dataclass(frozen=True)
class LqProblem:
    residuals: dict[str, Tensor]
    cost_breakdown: dict[str, Tensor]
    diagonal: Tensor
    first_offdiag: Tensor
    second_offdiag: Tensor
    gradient: Tensor

    @property
    def total_cost(self) -> Tensor:
        total = torch.zeros_like(self.cost_breakdown[RESIDUAL_FAMILIES[0]])
        for name in RESIDUAL_FAMILIES:
            total = total + self.cost_breakdown[name]
        return total

    def to_dense(self) -> tuple[Tensor, Tensor]:
        batch, nodes, state_dim = self.gradient.shape
        blocks = self.gradient.new_zeros(batch, nodes, nodes, state_dim, state_dim)
        node = torch.arange(nodes, device=self.gradient.device)
        edge = torch.arange(nodes - 1, device=self.gradient.device)
        second = torch.arange(nodes - 2, device=self.gradient.device)
        blocks[:, node, node] = self.diagonal
        blocks[:, edge, edge + 1] = self.first_offdiag
        blocks[:, edge + 1, edge] = self.first_offdiag.transpose(-1, -2)
        blocks[:, second, second + 2] = self.second_offdiag
        blocks[:, second + 2, second] = self.second_offdiag.transpose(-1, -2)
        dense = blocks.permute(0, 1, 3, 2, 4).reshape(
            batch, nodes * state_dim, nodes * state_dim
        )
        return dense, self.gradient.flatten(1)


def _state_identity(state: Tensor) -> Tensor:
    return torch.eye(18, dtype=state.dtype, device=state.device).view(1, 1, 18, 18)


def _wrap_angle(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _foot_kinematics(state: Tensor) -> tuple[Tensor, Tensor]:
    batch, nodes = state.shape[:2]
    flat = state.reshape(batch * nodes, 18)
    foot = go2_fk(flat[:, :3], flat[:, 3:6], flat[:, 6:]).foot_pos_w
    jacobian = complete_foot_jacobian(flat[:, :3], flat[:, 3:6], flat[:, 6:])
    return (
        foot.reshape(batch, nodes, 4, 3),
        jacobian.reshape(batch, nodes, 4, 3, 18),
    )


def _velocity_term(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> _LocalTerm:
    dt = float(cfg.runtime.dt)
    delta = state[:, 1:, :2] - state[:, :-1, :2]
    yaw = state[:, :-1, 5]
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    vx = (cosine * delta[..., 0] + sine * delta[..., 1]) / dt
    vy = (-sine * delta[..., 0] + cosine * delta[..., 1]) / dt
    linear_scale = math.sqrt(float(cfg.lq_cost.velocity_linear))
    yaw_scale = math.sqrt(float(cfg.lq_cost.velocity_yaw))
    linear = torch.stack((vx, vy), dim=-1)
    linear = linear_scale * (linear - context.command_body[:, None, :2])
    yaw_rate = _wrap_angle(state[:, 1:, 5] - state[:, :-1, 5]) / dt
    yaw_residual = yaw_scale * (yaw_rate - context.command_body[:, None, 2])
    residual = torch.cat((linear, yaw_residual[..., None]), dim=-1)

    batch, edges = yaw.shape
    jacobian0 = state.new_zeros(batch, edges, 3, 18)
    jacobian1 = state.new_zeros(batch, edges, 3, 18)
    rotation = torch.stack(
        (
            torch.stack((cosine, sine), dim=-1),
            torch.stack((-sine, cosine), dim=-1),
        ),
        dim=-2,
    )
    jacobian0[..., :2, :2] = -linear_scale * rotation / dt
    jacobian1[..., :2, :2] = linear_scale * rotation / dt
    jacobian0[..., 0, 5] = linear_scale * vy
    jacobian0[..., 1, 5] = -linear_scale * vx
    jacobian0[..., 2, 5] = -yaw_scale / dt
    jacobian1[..., 2, 5] = yaw_scale / dt
    return _LocalTerm(residual, (jacobian0, jacobian1))


def _posture_term(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> _LocalTerm:
    speed2 = context.command_body[:, :2].square().sum(dim=-1)
    velocity_scale2 = max(float(cfg.lq_cost.hold_velocity_scale) ** 2, 1.0e-12)
    height_delta = context.support_height[:, 1:] - context.support_height[:, :-1]
    roughness2 = height_delta.abs().amax(dim=1).square()
    roughness_scale2 = max(float(cfg.lq_cost.hold_roughness_scale) ** 2, 1.0e-12)
    hold = torch.exp(-speed2 / velocity_scale2) * torch.exp(
        -roughness2 / roughness_scale2
    )
    scale = torch.sqrt(hold * float(cfg.lq_cost.posture_joint))
    reference = state.new_tensor(cfg.gait.nominal_joint_pos).view(1, 1, 12)
    residual = scale[:, None, None] * (state[..., 6:] - reference)
    jacobian = state.new_zeros(*residual.shape, 18)
    selector = torch.eye(12, dtype=state.dtype, device=state.device)
    jacobian[..., 6:] = scale[:, None, None, None] * selector
    return _LocalTerm(residual, (jacobian,))


def _root_terms(
    state: Tensor,
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> tuple[_LocalTerm, ...]:
    root_scale = state.new_tensor(
        (
            math.sqrt(float(cfg.lq_cost.root_height)),
            math.sqrt(float(cfg.lq_cost.root_roll_pitch)),
            math.sqrt(float(cfg.lq_cost.root_roll_pitch)),
        )
    )
    root_reference = torch.stack(
        (
            context.support_height + float(cfg.loss_terms.posture_root_clearance),
            nominal.state[:, :1, 3].detach().expand(-1, state.shape[1]),
            nominal.state[:, :1, 4].detach().expand(-1, state.shape[1]),
        ),
        dim=-1,
    )
    root_value = torch.stack((state[..., 2], state[..., 3], state[..., 4]), dim=-1)
    root_residual = root_scale * (root_value - root_reference)
    root_jacobian = state.new_zeros(*root_residual.shape, 18)
    root_jacobian[..., 0, 2] = root_scale[0]
    root_jacobian[..., 1, 3] = root_scale[1]
    root_jacobian[..., 2, 4] = root_scale[2]

    corridor_scale = math.sqrt(float(cfg.lq_cost.root_corridor))
    corridor_reference = nominal.state.detach()[..., :2]
    corridor_residual = corridor_scale * (state[..., :2] - corridor_reference)
    corridor_jacobian = state.new_zeros(*corridor_residual.shape, 18)
    corridor_jacobian[..., 0, 0] = corridor_scale
    corridor_jacobian[..., 1, 1] = corridor_scale

    rate_scale = math.sqrt(float(cfg.lq_cost.root_rate)) / float(cfg.runtime.dt)
    rate_residual = rate_scale * (
        state[:, 1:, 2:5] - state[:, :-1, 2:5]
    )
    rate_jacobian0 = state.new_zeros(*rate_residual.shape, 18)
    rate_jacobian1 = state.new_zeros(*rate_residual.shape, 18)
    selector = torch.eye(3, dtype=state.dtype, device=state.device)
    rate_jacobian0[..., 2:5] = -rate_scale * selector
    rate_jacobian1[..., 2:5] = rate_scale * selector
    return (
        _LocalTerm(root_residual, (root_jacobian,)),
        _LocalTerm(corridor_residual, (corridor_jacobian,)),
        _LocalTerm(rate_residual, (rate_jacobian0, rate_jacobian1)),
    )


def _swing_terms(
    state: Tensor,
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
    foot: Tensor,
    foot_jacobian: Tensor,
) -> tuple[_LocalTerm, ...]:
    position_scale = math.sqrt(float(cfg.lq_cost.swing_position))
    node_mask = context.schedule.swing.to(state.dtype)[..., None]
    position_residual = position_scale * node_mask * (
        foot - nominal.foot_reference_w
    )
    position_jacobian = position_scale * node_mask[..., None] * foot_jacobian
    position = _LocalTerm(
        position_residual.reshape(*state.shape[:2], 12),
        (position_jacobian.reshape(*state.shape[:2], 12, 18),),
    )

    dt = float(cfg.runtime.dt)
    velocity_scale = math.sqrt(float(cfg.lq_cost.swing_velocity)) / dt
    edge_mask = context.schedule.swing_edge.to(state.dtype)[..., None]
    velocity_residual = velocity_scale * edge_mask * (
        (foot[:, 1:] - foot[:, :-1])
        - (nominal.foot_reference_w[:, 1:] - nominal.foot_reference_w[:, :-1])
    )
    jacobian0 = -velocity_scale * edge_mask[..., None] * foot_jacobian[:, :-1]
    jacobian1 = velocity_scale * edge_mask[..., None] * foot_jacobian[:, 1:]
    velocity = _LocalTerm(
        velocity_residual.reshape(state.shape[0], state.shape[1] - 1, 12),
        (
            jacobian0.reshape(state.shape[0], state.shape[1] - 1, 12, 18),
            jacobian1.reshape(state.shape[0], state.shape[1] - 1, 12, 18),
        ),
    )
    return position, velocity


def _touchdown_term(
    state: Tensor,
    context: LossContext,
    cfg: JointMpcRtiCfg,
    foot: Tensor,
    foot_jacobian: Tensor,
) -> _LocalTerm:
    touchdown_node = torch.cat(
        (
            torch.zeros_like(context.schedule.touchdown_edge[:, :1]),
            context.schedule.touchdown_edge,
        ),
        dim=1,
    )
    axis_scale = state.new_tensor(
        (
            math.sqrt(float(cfg.lq_cost.touchdown_xy)),
            math.sqrt(float(cfg.lq_cost.touchdown_xy)),
            math.sqrt(float(cfg.lq_cost.touchdown_z)),
        )
    )
    mask = touchdown_node.to(state.dtype)[..., None]
    residual = mask * axis_scale * (foot - context.touchdown_reference_w)
    jacobian = mask[..., None] * axis_scale.view(1, 1, 1, 3, 1) * foot_jacobian
    return _LocalTerm(
        residual.reshape(*state.shape[:2], 12),
        (jacobian.reshape(*state.shape[:2], 12, 18),),
    )


def _smooth_terms(state: Tensor, cfg: JointMpcRtiCfg) -> tuple[_LocalTerm, ...]:
    identity = _state_identity(state)
    first_scale = math.sqrt(float(cfg.lq_cost.smooth_first))
    first_residual = first_scale * (state[:, 1:] - state[:, :-1])
    first_jacobian0 = (-first_scale * identity).expand(state.shape[0], 30, -1, -1)
    first_jacobian1 = (first_scale * identity).expand(state.shape[0], 30, -1, -1)
    second_scale = math.sqrt(float(cfg.lq_cost.smooth_second))
    second_residual = second_scale * (
        state[:, 2:] - 2.0 * state[:, 1:-1] + state[:, :-2]
    )
    second_jacobian0 = (second_scale * identity).expand(state.shape[0], 29, -1, -1)
    second_jacobian1 = (-2.0 * second_scale * identity).expand(state.shape[0], 29, -1, -1)
    second_jacobian2 = (second_scale * identity).expand(state.shape[0], 29, -1, -1)
    return (
        _LocalTerm(first_residual, (first_jacobian0, first_jacobian1)),
        _LocalTerm(
            second_residual,
            (second_jacobian0, second_jacobian1, second_jacobian2),
        ),
    )


def _warm_term(state: Tensor, nominal: NominalTrajectory, cfg: JointMpcRtiCfg) -> _LocalTerm:
    scale = math.sqrt(float(cfg.lq_cost.warm))
    residual = scale * (state - nominal.rebased_state)
    jacobian = (scale * _state_identity(state)).expand(state.shape[0], 31, -1, -1)
    return _LocalTerm(residual, (jacobian,))


def _family_terms(
    state: Tensor,
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> dict[str, tuple[_LocalTerm, ...]]:
    value = torch.as_tensor(state)
    if value.ndim != 3 or value.shape[1:] != (31, 18):
        raise ValueError("state must have shape [B,31,18]")
    foot, foot_jacobian = _foot_kinematics(value)
    return {
        "velocity": (_velocity_term(value, context, cfg),),
        "posture": (_posture_term(value, context, cfg),),
        "root": _root_terms(value, nominal, context, cfg),
        "swing": _swing_terms(value, nominal, context, cfg, foot, foot_jacobian),
        "touchdown": (
            _touchdown_term(value, context, cfg, foot, foot_jacobian),
        ),
        "smooth": _smooth_terms(value, cfg),
        "warm": (_warm_term(value, nominal, cfg),),
        "slack": (),
    }


def _flatten_terms(state: Tensor, terms: tuple[_LocalTerm, ...]) -> Tensor:
    if not terms:
        return state.new_zeros(state.shape[0], 0)
    return torch.cat(tuple(term.residual.flatten(1) for term in terms), dim=1)


def lq_residuals(
    state: Tensor,
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> dict[str, Tensor]:
    terms = _family_terms(state, nominal, context, cfg)
    return {name: _flatten_terms(state, terms[name]) for name in RESIDUAL_FAMILIES}


def _accumulate_term(
    term: _LocalTerm,
    diagonal: Tensor,
    first_offdiag: Tensor,
    second_offdiag: Tensor,
    gradient: Tensor,
) -> None:
    residual = term.residual
    jacobians = term.jacobians
    arity = len(jacobians)
    if arity not in (1, 2, 3):
        raise ValueError("local LQ terms must touch one, two, or three nodes")
    term_nodes = int(residual.shape[1])
    for offset, jacobian in enumerate(jacobians):
        gradient[:, offset : offset + term_nodes] += torch.einsum(
            "bnri,bnr->bni", jacobian, residual
        )
        diagonal[:, offset : offset + term_nodes] += torch.einsum(
            "bnri,bnrj->bnij", jacobian, jacobian
        )
    if arity >= 2:
        first_offdiag[:, :term_nodes] += torch.einsum(
            "bnri,bnrj->bnij", jacobians[0], jacobians[1]
        )
    if arity == 3:
        first_offdiag[:, 1 : 1 + term_nodes] += torch.einsum(
            "bnri,bnrj->bnij", jacobians[1], jacobians[2]
        )
        second_offdiag[:, :term_nodes] += torch.einsum(
            "bnri,bnrj->bnij", jacobians[0], jacobians[2]
        )


def build_lq_problem(
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> LqProblem:
    state = torch.as_tensor(nominal.state)
    terms = _family_terms(state, nominal, context, cfg)
    residuals = {
        name: _flatten_terms(state, terms[name]) for name in RESIDUAL_FAMILIES
    }
    cost_breakdown = {
        name: 0.5 * residuals[name].square().sum(dim=1)
        for name in RESIDUAL_FAMILIES
    }
    batch = int(state.shape[0])
    diagonal = state.new_zeros(batch, 31, 18, 18)
    first_offdiag = state.new_zeros(batch, 30, 18, 18)
    second_offdiag = state.new_zeros(batch, 29, 18, 18)
    gradient = state.new_zeros(batch, 31, 18)
    for name in RESIDUAL_FAMILIES:
        for term in terms[name]:
            _accumulate_term(
                term, diagonal, first_offdiag, second_offdiag, gradient
            )
    regularization = float(cfg.solver.regularization)
    if regularization:
        diagonal = diagonal + regularization * _state_identity(state)
    return LqProblem(
        residuals=residuals,
        cost_breakdown=cost_breakdown,
        diagonal=diagonal,
        first_offdiag=first_offdiag,
        second_offdiag=second_offdiag,
        gradient=gradient,
    )


__all__ = [
    "LqProblem",
    "RESIDUAL_FAMILIES",
    "build_lq_problem",
    "lq_residuals",
]
