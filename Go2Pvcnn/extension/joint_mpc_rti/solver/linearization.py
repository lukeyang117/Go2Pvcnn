"""Gauss-Newton linearization directly in the H30 state trajectory Z."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import LossContext, weighted_trajectory_residual
from extension.joint_mpc_rti.losses.terrain import effective_foot_surface_height
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.model.go2_kinematics import complete_foot_jacobian, go2_fk
from extension.joint_mpc_rti.solver.trajectory_qp import TrajectoryQp, trajectory_bounds
from extension.joint_mpc_rti.terrain.query import query_world
from extension.joint_mpc_rti.types import JointMpcTerrainField


def published_kinematic_jacobian(state: Tensor, schedule: FixedTrotSchedule) -> Tensor:
    """Return two stance-XY and two swing-Z Jacobian rows at published x1."""
    nominal = torch.as_tensor(state)
    if nominal.ndim != 3 or nominal.shape[1:] != (31, 18):
        raise ValueError("state must have shape [B,31,18]")
    stance = torch.as_tensor(schedule.stance, dtype=torch.bool, device=nominal.device)
    if stance.shape != (nominal.shape[0], 31, 4):
        raise ValueError("schedule stance must have shape [B,31,4]")
    stance_index = torch.topk(stance[:, 1].to(torch.int64), k=2, dim=1).indices
    swing_index = torch.topk((~stance[:, 1]).to(torch.int64), k=2, dim=1).indices
    jacobian = complete_foot_jacobian(
        nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
    )
    selected_stance = torch.gather(
        jacobian,
        1,
        stance_index[..., None, None].expand(-1, -1, 3, 18),
    )
    selected_swing = torch.gather(
        jacobian,
        1,
        swing_index[..., None, None].expand(-1, -1, 3, 18),
    )
    return torch.cat(
        (
            selected_stance[..., :2, :].reshape(nominal.shape[0], 4, 18),
            selected_swing[..., 2, :],
        ),
        dim=1,
    )


def published_kinematic_target(
    state: Tensor,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    """Return stance-XY corrections and nonnegative swing-Z floor corrections."""
    nominal = torch.as_tensor(state)
    anchor = torch.as_tensor(
        context.stance_anchor_w, dtype=nominal.dtype, device=nominal.device
    )
    if anchor.shape != (nominal.shape[0], 31, 4, 3):
        raise ValueError("stance_anchor_w must have shape [B,31,4,3]")
    stance = torch.as_tensor(
        context.schedule.stance, dtype=torch.bool, device=nominal.device
    )
    stance_index = torch.topk(stance[:, 1].to(torch.int64), k=2, dim=1).indices
    swing_index = torch.topk((~stance[:, 1]).to(torch.int64), k=2, dim=1).indices
    foot = go2_fk(
        nominal[:, 1, :3], nominal[:, 1, 3:6], nominal[:, 1, 6:]
    ).foot_pos_w
    gather_index = stance_index[..., None].expand(-1, -1, 3)
    onset = ~stance[:, 0] & stance[:, 1]
    selected_anchor = torch.gather(anchor[:, 1], 1, gather_index)
    selected_foot = torch.gather(foot, 1, gather_index)
    selected_onset = torch.gather(onset, 1, stance_index)
    correction_xy = selected_anchor[..., :2] - selected_foot[..., :2]
    correction_xy = torch.where(
        selected_onset[..., None], torch.zeros_like(correction_xy), correction_xy
    )
    correction_xy = correction_xy.reshape(nominal.shape[0], 4)

    query = query_world(context.terrain, foot)
    surface = effective_foot_surface_height(
        query.height_w,
        query.small_occupancy,
        query.large_occupancy,
        query.small_propagated_height,
        stance=stance[:, 1],
        h_wall=float(cfg.terrain.h_wall),
    ).to(nominal.dtype)
    selected_surface = torch.gather(surface, 1, swing_index)
    selected_swing_z = torch.gather(foot[..., 2], 1, swing_index)
    safe_z = (
        selected_surface
        + float(cfg.gait.foot_contact_offset)
        + float(cfg.solver.published_swing_clearance_buffer)
    )
    swing_correction_z = (safe_z - selected_swing_z).clamp_min(0.0)
    return torch.cat((correction_xy, swing_correction_z), dim=1)


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
        support_jacobian=published_kinematic_jacobian(nominal, context.schedule),
        support_target=published_kinematic_target(nominal, context, cfg),
    )


__all__ = [
    "linearize_trajectory",
    "published_kinematic_jacobian",
    "published_kinematic_target",
]
