"""Graph-safe fixed-size SPD solves for batched RTI kernels."""

from __future__ import annotations

import torch
from torch import Tensor


def fixed_spd_solve(matrix: Tensor, rhs: Tensor) -> Tensor:
    """Solve batched SPD systems without dispatching to cuSolver or MAGMA."""
    system = torch.as_tensor(matrix)
    right = torch.as_tensor(rhs, dtype=system.dtype, device=system.device)
    if system.ndim < 2 or system.shape[-1] != system.shape[-2]:
        raise ValueError("matrix must have shape [..., N, N]")
    dimension = int(system.shape[-1])
    if right.shape[:-2] != system.shape[:-2] or right.shape[-2] != dimension:
        raise ValueError("rhs must have shape [..., N, R]")
    if system.is_cuda:
        from extension.joint_mpc_rti.solver.fixed_spd_triton import fixed_spd_solve_cuda

        return fixed_spd_solve_cuda(system.contiguous(), right.contiguous())

    rows: list[Tensor] = []
    for row_index in range(dimension):
        entries: list[Tensor] = []
        for column_index in range(dimension):
            if column_index < row_index:
                product = torch.zeros_like(system[..., row_index, column_index])
                for inner_index in range(column_index):
                    product = product + entries[inner_index] * rows[column_index][..., inner_index]
                value = (
                    system[..., row_index, column_index] - product
                ) / rows[column_index][..., column_index]
            elif column_index == row_index:
                diagonal = torch.zeros_like(system[..., row_index, row_index])
                for inner_index in range(row_index):
                    diagonal = diagonal + entries[inner_index].square()
                value = torch.sqrt(
                    (system[..., row_index, row_index] - diagonal).clamp_min(
                        torch.finfo(system.dtype).eps
                    )
                )
            else:
                value = torch.zeros_like(system[..., row_index, column_index])
            entries.append(value)
        rows.append(torch.stack(entries, dim=-1))
    lower = torch.stack(rows, dim=-2)

    forward_rows: list[Tensor] = []
    for row_index in range(dimension):
        residual = right[..., row_index, :]
        for column_index in range(row_index):
            residual = residual - lower[..., row_index, column_index, None] * forward_rows[column_index]
        forward_rows.append(residual / lower[..., row_index, row_index, None])

    solution_reverse: list[Tensor] = []
    for row_index in range(dimension - 1, -1, -1):
        residual = forward_rows[row_index]
        for column_index in range(row_index + 1, dimension):
            solved = solution_reverse[dimension - 1 - column_index]
            residual = residual - lower[..., column_index, row_index, None] * solved
        solution_reverse.append(residual / lower[..., row_index, row_index, None])
    return torch.stack(solution_reverse[::-1], dim=-2)


__all__ = ["fixed_spd_solve"]
