"""Fixed-shape associative compositions used by the H30 trajectory solver."""

from __future__ import annotations

import torch
from torch import Tensor


ConditionalValueFactor = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def combine_conditional_value_factors(
    left: ConditionalValueFactor,
    right: ConditionalValueFactor,
) -> ConditionalValueFactor:
    """Compose adjacent conditional value factors in temporal order."""
    matrix_a_left, vector_c_left, matrix_c_left, vector_p_left, matrix_p_left = left
    matrix_a_right, vector_c_right, matrix_c_right, vector_p_right, matrix_p_right = right
    dimension = int(matrix_a_left.shape[-1])
    identity = torch.eye(
        dimension,
        dtype=matrix_a_left.dtype,
        device=matrix_a_left.device,
    )
    coupling = identity + matrix_c_left @ matrix_p_right
    right_elimination = torch.linalg.solve(
        coupling.transpose(-1, -2), matrix_a_right.transpose(-1, -2)
    ).transpose(-1, -2)
    left_elimination = torch.linalg.solve(coupling, matrix_a_left).transpose(-1, -2)
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


def _compose_affine(left: tuple[Tensor, Tensor], right: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    matrix_left, bias_left = left
    matrix_right, bias_right = right
    return matrix_right @ matrix_left, (matrix_right @ bias_left.unsqueeze(-1)).squeeze(-1) + bias_right


def _interleave_prefix(
    previous: tuple[Tensor, Tensor],
    source: tuple[Tensor, Tensor],
    stride: int,
) -> tuple[Tensor, Tensor]:
    matrix, bias = previous
    left = (matrix[:, :-stride], bias[:, :-stride])
    right = (source[0][:, stride:], source[1][:, stride:])
    composed = _compose_affine(left, right)
    return (
        torch.cat((matrix[:, :stride], composed[0]), dim=1),
        torch.cat((bias[:, :stride], composed[1]), dim=1),
    )


def affine_scan(matrix: Tensor, bias: Tensor, initial: Tensor | None = None) -> Tensor:
    """Apply a fixed H16 affine prefix scan without the generic HOP scan."""
    matrix_tensor = torch.as_tensor(matrix)
    bias_tensor = torch.as_tensor(bias, dtype=matrix_tensor.dtype, device=matrix_tensor.device)
    if matrix_tensor.ndim != 4 or matrix_tensor.shape[1] != 16:
        raise ValueError("matrix must have fixed shape [B,16,N,N]")
    if bias_tensor.shape != matrix_tensor.shape[:-1]:
        raise ValueError("bias must have shape [B,16,N]")
    level_1 = _interleave_prefix((matrix_tensor, bias_tensor), (matrix_tensor, bias_tensor), 1)
    level_2 = _interleave_prefix(level_1, level_1, 2)
    level_3 = _interleave_prefix(level_2, level_2, 4)
    level_4 = _interleave_prefix(level_3, level_3, 8)
    matrix_prefix, bias_prefix = level_4
    if initial is None:
        return bias_prefix
    initial_tensor = torch.as_tensor(initial, dtype=matrix_tensor.dtype, device=matrix_tensor.device)
    if initial_tensor.shape != bias_tensor[:, 0].shape:
        raise ValueError("initial must have shape [B,N]")
    return (matrix_prefix @ initial_tensor[:, None, :, None]).squeeze(-1) + bias_prefix


__all__ = [
    "ConditionalValueFactor",
    "affine_scan",
    "combine_conditional_value_factors",
]
