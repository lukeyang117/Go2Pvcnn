"""Fixed H30/32 associative solve for the direct-state trajectory QP."""

from __future__ import annotations

import torch
from torch import Tensor

from .associative_scan import ConditionalValueFactor, combine_conditional_value_factors
from .trajectory_qp import ActiveConstraints, ActiveSetSolution, TrajectoryQp, refine_active_set


STATE_DIM = 18
SEPARATOR_DIM = 2 * STATE_DIM
INTERVALS = 30
PADDED_INTERVALS = 32


def pad_h30_factors(
    factors: ConditionalValueFactor,
) -> tuple[ConditionalValueFactor, Tensor]:
    """Append two identity/no-cost intervals to a time-major H30 factor tuple."""
    if any(int(value.shape[0]) != INTERVALS for value in factors):
        raise ValueError("trajectory factors must contain exactly 30 intervals")
    matrix_a, vector_c, matrix_c, vector_p, matrix_p = factors
    dimension = int(matrix_a.shape[-1])
    identity = torch.eye(dimension, dtype=matrix_a.dtype, device=matrix_a.device)
    identity = identity.expand(2, *matrix_a.shape[1:-2], dimension, dimension)

    def append_zeros(value: Tensor) -> Tensor:
        return torch.cat((value, value.new_zeros(2, *value.shape[1:])), dim=0)

    padded = (
        torch.cat((matrix_a, identity), dim=0),
        append_zeros(vector_c),
        append_zeros(matrix_c),
        append_zeros(vector_p),
        append_zeros(matrix_p),
    )
    valid = torch.arange(PADDED_INTERVALS, device=matrix_a.device) < INTERVALS
    return padded, valid


def _combine_pairs(factors: ConditionalValueFactor) -> ConditionalValueFactor:
    left = tuple(value[0::2] for value in factors)
    right = tuple(value[1::2] for value in factors)
    return combine_conditional_value_factors(left, right)


def _fixed_five_level_tree(factors: ConditionalValueFactor) -> tuple[ConditionalValueFactor, ...]:
    combine_level_1 = _combine_pairs(factors)
    combine_level_2 = _combine_pairs(combine_level_1)
    combine_level_3 = _combine_pairs(combine_level_2)
    combine_level_4 = _combine_pairs(combine_level_3)
    combine_level_5 = _combine_pairs(combine_level_4)
    return factors, combine_level_1, combine_level_2, combine_level_3, combine_level_4, combine_level_5


def _active_control_parameterization(
    qp: TrajectoryQp,
    active: ActiveConstraints,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return u=F y+f+N v for local box and velocity equalities."""
    batch = qp.batch_size
    dtype = qp.gradient.dtype
    device = qp.gradient.device
    fixed = active.box_mask[:, 1:].clone()
    affine = torch.zeros(batch, INTERVALS, STATE_DIM, SEPARATOR_DIM, dtype=dtype, device=device)
    offset = torch.zeros(batch, INTERVALS, STATE_DIM, dtype=dtype, device=device)
    box_target = torch.where(active.box_low[:, 1:], qp.lower[:, 1:], qp.upper[:, 1:])
    offset = torch.where(fixed, box_target, offset)

    joint_box = active.box_mask[..., 6:]
    joint_box_target = torch.where(active.box_low[..., 6:], qp.lower[..., 6:], qp.upper[..., 6:])
    redundant_velocity = joint_box[:, :-1] & joint_box[:, 1:]
    velocity = active.velocity_mask & ~redundant_velocity
    velocity_target = torch.where(active.velocity_low, qp.joint_difference_lower, qp.joint_difference_upper)
    edge_delta = torch.where(velocity, velocity_target, torch.zeros_like(velocity_target))
    cumulative = torch.cat(
        (edge_delta.new_zeros(batch, 1, 12), edge_delta.cumsum(dim=1)), dim=1
    )
    node_index = torch.arange(1, 31, device=device).view(1, 30, 1)
    reset_index = torch.where(~velocity, node_index, torch.zeros_like(node_index))
    segment_start = torch.cat(
        (torch.zeros(batch, 1, 12, dtype=torch.long, device=device), reset_index.expand(batch, -1, 12)),
        dim=1,
    ).cummax(dim=1).values
    potential = cumulative - cumulative.gather(1, segment_start)
    segment = torch.cat(
        (
            torch.zeros(batch, 1, 12, dtype=torch.long, device=device),
            (~velocity).to(torch.long).cumsum(dim=1),
        ),
        dim=1,
    )
    segment_by_joint = segment.permute(0, 2, 1)
    anchor_base = torch.where(joint_box, joint_box_target - potential, torch.zeros_like(potential))
    base_sum = torch.zeros(batch, 12, 31, dtype=dtype, device=device).scatter_add(
        2, segment_by_joint, anchor_base.permute(0, 2, 1)
    )
    anchor_count = torch.zeros(batch, 12, 31, dtype=dtype, device=device).scatter_add(
        2, segment_by_joint, joint_box.to(dtype).permute(0, 2, 1)
    )
    node_count = anchor_count.gather(2, segment_by_joint).permute(0, 2, 1)
    node_base = (
        base_sum.gather(2, segment_by_joint)
        / anchor_count.gather(2, segment_by_joint).clamp_min(1.0)
    ).permute(0, 2, 1)
    component_fixed = node_count > 0
    propagated_target = node_base + potential
    fixed[..., 6:] = component_fixed[:, 1:]
    offset[..., 6:] = torch.where(component_fixed[:, 1:], propagated_target[:, 1:], offset[..., 6:])
    velocity_only = velocity & ~component_fixed[:, 1:]
    fixed[..., 6:] |= velocity_only
    separator_joint = torch.arange(12, device=device) + STATE_DIM + 6
    affine[..., 6:, separator_joint] = torch.diag_embed(velocity_only.to(dtype))
    offset[..., 6:] = torch.where(velocity_only, velocity_target, offset[..., 6:])
    free = (~fixed).to(dtype)
    return affine, offset, free


def _trajectory_factors(qp: TrajectoryQp, active: ActiveConstraints) -> ConditionalValueFactor:
    if qp.nodes != 31 or int(qp.gradient.shape[-1]) != STATE_DIM:
        raise ValueError("trajectory scan requires [B,31,18]")
    batch = qp.batch_size
    dtype = qp.gradient.dtype
    device = qp.gradient.device
    affine_control, control_offset, free = _active_control_parameterization(qp, active)
    identity = torch.eye(STATE_DIM, dtype=dtype, device=device)
    dynamics_a = torch.zeros(batch, INTERVALS, SEPARATOR_DIM, SEPARATOR_DIM, dtype=dtype, device=device)
    dynamics_a[..., :STATE_DIM, STATE_DIM:] = identity
    dynamics_b = torch.zeros(batch, INTERVALS, SEPARATOR_DIM, STATE_DIM, dtype=dtype, device=device)
    dynamics_b[..., STATE_DIM:, :] = identity

    cross = torch.zeros(batch, INTERVALS, SEPARATOR_DIM, STATE_DIM, dtype=dtype, device=device)
    cross[..., STATE_DIM:, :] = qp.first_offdiag
    cross[:, 1:, :STATE_DIM, :] = qp.second_offdiag
    control_hessian = qp.diagonal[:, 1:]
    control_gradient = qp.gradient[:, 1:]
    free_matrix = torch.diag_embed(free)
    fixed_matrix = torch.diag_embed(1.0 - free)

    transformed_a = dynamics_a + dynamics_b @ affine_control
    transformed_b = dynamics_b @ free_matrix
    transformed_c = (dynamics_b @ control_offset.unsqueeze(-1)).squeeze(-1)
    transformed_q = (
        affine_control.transpose(-1, -2) @ control_hessian @ affine_control
        + cross @ affine_control
        + (cross @ affine_control).transpose(-1, -2)
    )
    transformed_linear = (
        affine_control.transpose(-1, -2)
        @ (control_hessian @ control_offset.unsqueeze(-1) + control_gradient.unsqueeze(-1))
    ).squeeze(-1) + (cross @ control_offset.unsqueeze(-1)).squeeze(-1)
    transformed_r = free_matrix @ control_hessian @ free_matrix + fixed_matrix
    transformed_control_linear = (
        free_matrix @ (control_hessian @ control_offset.unsqueeze(-1) + control_gradient.unsqueeze(-1))
    ).squeeze(-1)
    transformed_cross = (affine_control.transpose(-1, -2) @ control_hessian + cross) @ free_matrix

    solve_b = torch.linalg.solve(transformed_r, transformed_b.transpose(-1, -2)).transpose(-1, -2)
    solve_m = torch.linalg.solve(transformed_r, transformed_cross.transpose(-1, -2)).transpose(-1, -2)
    matrix_a = transformed_a - solve_b @ transformed_cross.transpose(-1, -2)
    vector_c = transformed_c - (solve_b @ transformed_control_linear.unsqueeze(-1)).squeeze(-1)
    matrix_c = solve_b @ transformed_b.transpose(-1, -2)
    vector_p = transformed_linear - (solve_m @ transformed_control_linear.unsqueeze(-1)).squeeze(-1)
    matrix_p = transformed_q - solve_m @ transformed_cross.transpose(-1, -2)
    return tuple(value.movedim(1, 0) for value in (matrix_a, vector_c, matrix_c, vector_p, matrix_p))


def _split_boundaries(
    left: ConditionalValueFactor,
    right: ConditionalValueFactor,
    state_left: Tensor,
    costate_right: Tensor,
) -> tuple[Tensor, Tensor]:
    a_left, c_left, c_matrix_left, _, _ = left
    a_right, _, _, p_right, p_matrix_right = right
    identity = torch.eye(a_left.shape[-1], dtype=a_left.dtype, device=a_left.device)
    rhs = (
        (a_left @ state_left.unsqueeze(-1)).squeeze(-1)
        + c_left
        - (
            c_matrix_left
            @ (p_right + (a_right.transpose(-1, -2) @ costate_right.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
        ).squeeze(-1)
    )
    state_middle = torch.linalg.solve(identity + c_matrix_left @ p_matrix_right, rhs.unsqueeze(-1)).squeeze(-1)
    costate_middle = (
        a_right.transpose(-1, -2) @ costate_right.unsqueeze(-1)
    ).squeeze(-1) + p_right + (p_matrix_right @ state_middle.unsqueeze(-1)).squeeze(-1)
    return state_middle, costate_middle


def _expand_boundaries(
    child_factors: ConditionalValueFactor,
    state_left: Tensor,
    costate_right: Tensor,
) -> tuple[Tensor, Tensor]:
    left = tuple(value[0::2] for value in child_factors)
    right = tuple(value[1::2] for value in child_factors)
    state_middle, costate_middle = _split_boundaries(left, right, state_left, costate_right)
    state_children = torch.stack((state_left, state_middle), dim=1).flatten(0, 1)
    costate_children = torch.stack((costate_middle, costate_right), dim=1).flatten(0, 1)
    return state_children, costate_children


def _recover_direction(levels: tuple[ConditionalValueFactor, ...], batch: int) -> Tensor:
    root = levels[-1]
    state_left = root[0].new_zeros(1, batch, SEPARATOR_DIM)
    costate_right = root[0].new_zeros(1, batch, SEPARATOR_DIM)
    for child_factors in reversed(levels[:-1]):
        state_left, costate_right = _expand_boundaries(child_factors, state_left, costate_right)
    final_state = (
        levels[0][0][-1] @ state_left[-1].unsqueeze(-1)
    ).squeeze(-1) + levels[0][1][-1] - (
        levels[0][2][-1] @ costate_right[-1].unsqueeze(-1)
    ).squeeze(-1)
    boundaries = torch.cat((state_left, final_state.unsqueeze(0)), dim=0)
    return boundaries[:31, :, STATE_DIM:].movedim(0, 1)


def solve_active_trajectory_qp_scan(qp: TrajectoryQp, active: ActiveConstraints) -> Tensor:
    factors, _ = pad_h30_factors(_trajectory_factors(qp, active))
    levels = _fixed_five_level_tree(factors)
    return _recover_direction(levels, qp.batch_size)


def solve_trajectory_qp_scan(qp: TrajectoryQp) -> ActiveSetSolution:
    return refine_active_set(qp, solve_active_trajectory_qp_scan, refinements=2)


__all__ = [
    "pad_h30_factors",
    "solve_active_trajectory_qp_scan",
    "solve_trajectory_qp_scan",
]
