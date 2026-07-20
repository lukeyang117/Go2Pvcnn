"""Gauss-Newton linearization directly in the H30 state trajectory Z."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import LossContext, weighted_trajectory_residual
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.solver.trajectory_qp import TrajectoryQp, trajectory_bounds
from extension.joint_mpc_rti.types import JointMpcTerrainField


def _single_residual(
    state: Tensor,
    command: Tensor,
    touchdown: Tensor,
    phase: Tensor,
    swing: Tensor,
    stance: Tensor,
    swing_tau: Tensor,
    terrain_tensors: tuple[Tensor, ...],
    stance_anchor: Tensor,
    support_height: Tensor,
    *,
    resolution: float,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    (
        height,
        semantic,
        small_distance,
        large_distance,
        small_gradient,
        large_gradient,
        valid_mask,
        origin,
        yaw,
        timestamp,
        version,
        small_occupancy,
        large_occupancy,
        small_propagated_height,
        large_propagated_height,
        small_occupancy_gradient,
        large_occupancy_gradient,
    ) = terrain_tensors
    field = JointMpcTerrainField(
        height_w=height.unsqueeze(0),
        semantic_id=semantic.unsqueeze(0),
        small_distance_m=small_distance.unsqueeze(0),
        large_distance_m=large_distance.unsqueeze(0),
        small_gradient_xy=small_gradient.unsqueeze(0),
        large_gradient_xy=large_gradient.unsqueeze(0),
        valid_mask=valid_mask.unsqueeze(0),
        origin_w=origin.unsqueeze(0),
        yaw_w=yaw.unsqueeze(0),
        timestamp=timestamp.unsqueeze(0),
        version=version.unsqueeze(0),
        resolution=resolution,
        small_occupancy=small_occupancy.unsqueeze(0),
        large_occupancy=large_occupancy.unsqueeze(0),
        small_propagated_height=small_propagated_height.unsqueeze(0),
        large_propagated_height=large_propagated_height.unsqueeze(0),
        small_occupancy_gradient_xy=small_occupancy_gradient.unsqueeze(0),
        large_occupancy_gradient_xy=large_occupancy_gradient.unsqueeze(0),
    )
    schedule = FixedTrotSchedule(
        phase=phase.unsqueeze(0),
        swing=swing.unsqueeze(0),
        stance=stance.unsqueeze(0),
        swing_tau=swing_tau.unsqueeze(0),
    )
    context = LossContext(
        command_body=command.unsqueeze(0),
        touchdown_reference_w=touchdown.unsqueeze(0),
        schedule=schedule,
        terrain=field,
        stance_anchor_w=stance_anchor.unsqueeze(0),
        support_height=support_height.unsqueeze(0),
    )
    return weighted_trajectory_residual(state.unsqueeze(0), context, cfg)[0]


def _terrain_tensor_tuple(field: JointMpcTerrainField) -> tuple[Tensor, ...]:
    if any(
        value is None
        for value in (
            field.small_occupancy,
            field.large_occupancy,
            field.small_propagated_height,
            field.large_propagated_height,
            field.small_occupancy_gradient_xy,
            field.large_occupancy_gradient_xy,
        )
    ):
        raise ValueError("trajectory linearization requires populated soft semantic fields")
    return (
        field.height_w,
        field.semantic_id,
        field.small_distance_m,
        field.large_distance_m,
        field.small_gradient_xy,
        field.large_gradient_xy,
        field.valid_mask,
        field.origin_w,
        field.yaw_w,
        field.timestamp,
        field.version,
        field.small_occupancy,
        field.large_occupancy,
        field.small_propagated_height,
        field.large_propagated_height,
        field.small_occupancy_gradient_xy,
        field.large_occupancy_gradient_xy,
    )


def linearize_trajectory(state: Tensor, context: LossContext, cfg: JointMpcRtiCfg) -> TrajectoryQp:
    """Build block-pentadiagonal GGN bands and approved trajectory bounds."""
    nominal = torch.as_tensor(state)
    if nominal.ndim != 3 or nominal.shape[1:] != (31, 18):
        raise ValueError("state must have shape [B,31,18]")
    terrain = _terrain_tensor_tuple(context.terrain)
    single = lambda *args: _single_residual(
        *args,
        resolution=float(context.terrain.resolution),
        cfg=cfg,
    )
    jacobian_fn = torch.func.jacrev(single, argnums=0)
    jacobian = torch.func.vmap(
        jacobian_fn,
        in_dims=(0, 0, 0, 0, 0, 0, 0, tuple(0 for _ in terrain), 0, 0),
    )(
        nominal,
        context.command_body,
        context.touchdown_reference_w,
        context.schedule.phase,
        context.schedule.swing,
        context.schedule.stance,
        context.schedule.swing_tau,
        terrain,
        context.stance_anchor_w,
        context.support_height,
    )
    residual = weighted_trajectory_residual(nominal, context, cfg)
    gradient = torch.einsum("brki,br->bki", jacobian, residual)
    diagonal = torch.einsum("brki,brkj->bkij", jacobian, jacobian)
    first_offdiag = torch.einsum("brki,brkj->bkij", jacobian[:, :, :-1], jacobian[:, :, 1:])
    second_offdiag = torch.einsum("brki,brkj->bkij", jacobian[:, :, :-2], jacobian[:, :, 2:])
    identity = torch.eye(18, dtype=nominal.dtype, device=nominal.device).view(1, 1, 18, 18)
    diagonal = diagonal + float(cfg.solver.regularization) * identity
    lower, upper, difference_lower, difference_upper = trajectory_bounds(nominal, cfg)
    return TrajectoryQp(
        diagonal=diagonal,
        first_offdiag=first_offdiag,
        second_offdiag=second_offdiag,
        gradient=gradient,
        lower=lower,
        upper=upper,
        joint_difference_lower=difference_lower,
        joint_difference_upper=difference_upper,
    )


__all__ = ["linearize_trajectory"]
