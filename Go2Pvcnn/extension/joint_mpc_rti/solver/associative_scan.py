"""Fixed-shape associative compositions used by the H30 trajectory solver."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve
from extension.joint_mpc_rti.tensor_constants import constant_like


ConditionalValueFactor = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def combine_conditional_value_factors(
    left: ConditionalValueFactor,
    right: ConditionalValueFactor,
) -> ConditionalValueFactor:
    """Compose adjacent conditional value factors in temporal order."""
    matrix_a_left, vector_c_left, matrix_c_left, vector_p_left, matrix_p_left = left
    matrix_a_right, vector_c_right, matrix_c_right, vector_p_right, matrix_p_right = right
    dimension = int(matrix_a_left.shape[-1])
    identity = constant_like(
        matrix_a_left,
        f"conditional_factor_identity_{dimension}",
        tuple(
            tuple(1.0 if row == column else 0.0 for column in range(dimension))
            for row in range(dimension)
        ),
    )
    coupling = identity + matrix_c_left @ matrix_p_right
    right_elimination = fixed_general_solve(
        coupling.transpose(-1, -2), matrix_a_right.transpose(-1, -2)
    ).transpose(-1, -2)
    left_elimination = fixed_general_solve(coupling, matrix_a_left).transpose(-1, -2)
    matrix_a = right_elimination @ matrix_a_left
    vector_c = (
        right_elimination
        @ (vector_c_left - (matrix_c_left @ vector_p_right.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
    ).squeeze(-1) + vector_c_right
    matrix_c = right_elimination @ matrix_c_left @ matrix_a_right.transpose(-1, -2) + matrix_c_right
    vector_p = (
        left_elimination
        @ (vector_p_right + (matrix_p_right @ vector_c_left.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
    ).squeeze(-1) + vector_p_left
    matrix_p = left_elimination @ matrix_p_right @ matrix_a_left + matrix_p_left
    return matrix_a, vector_c, matrix_c, vector_p, matrix_p


__all__ = [
    "ConditionalValueFactor",
    "combine_conditional_value_factors",
]
