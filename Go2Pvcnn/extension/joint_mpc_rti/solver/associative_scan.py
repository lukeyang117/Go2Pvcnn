"""Temporal associative scans used by GPU MPC kernels."""

from __future__ import annotations

import torch
from torch import Tensor
from torch._higher_order_ops.associative_scan import associative_scan


def _compose_affine(left: tuple[Tensor, Tensor], right: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    matrix_left, bias_left = left
    matrix_right, bias_right = right
    matrix = torch.matmul(matrix_right, matrix_left)
    bias = torch.matmul(matrix_right, bias_left.unsqueeze(-1)).squeeze(-1) + bias_right
    return matrix, bias


def affine_scan(matrix: Tensor, bias: Tensor, initial: Tensor | None = None) -> Tensor:
    """Apply ``x[k+1] = A[k] x[k] + b[k]`` with a temporal associative scan."""
    matrix_tensor = torch.as_tensor(matrix)
    bias_tensor = torch.as_tensor(bias, dtype=matrix_tensor.dtype, device=matrix_tensor.device)
    if matrix_tensor.ndim != 4 or int(matrix_tensor.shape[-1]) != int(matrix_tensor.shape[-2]):
        raise ValueError("matrix must have shape [B,H,N,N]")
    if bias_tensor.shape != matrix_tensor.shape[:-1]:
        raise ValueError("bias must have shape [B,H,N]")
    matrix_prefix, bias_prefix = associative_scan(
        _compose_affine,
        (matrix_tensor, bias_tensor),
        dim=1,
        combine_mode="generic",
    )
    if initial is None:
        return bias_prefix
    initial_tensor = torch.as_tensor(initial, dtype=matrix_tensor.dtype, device=matrix_tensor.device)
    if initial_tensor.shape != bias_tensor[:, 0].shape:
        raise ValueError("initial must have shape [B,N]")
    return torch.matmul(matrix_prefix, initial_tensor[:, None, :, None]).squeeze(-1) + bias_prefix


__all__ = ["affine_scan"]
