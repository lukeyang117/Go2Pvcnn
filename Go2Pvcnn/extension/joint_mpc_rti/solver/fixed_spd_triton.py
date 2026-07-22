"""Single-kernel batched SPD solve for fixed-size CUDA graph workloads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _fixed_spd_solve_kernel(
    matrix_ptr,
    rhs_ptr,
    output_ptr,
    systems,
    N: tl.constexpr,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    system_id = tl.program_id(0)
    row = tl.arange(0, BLOCK_N)[:, None]
    column = tl.arange(0, BLOCK_N)[None, :]
    matrix_offset = system_id * N * N + row * N + column
    matrix = tl.load(
        matrix_ptr + matrix_offset,
        mask=(system_id < systems) & (row < N) & (column < N),
        other=0.0,
    ).to(tl.float32)
    lower = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    epsilon = 1.0e-7
    for index in tl.static_range(0, N):
        lower_row = tl.sum(tl.where(row == index, lower, 0.0), axis=0)
        matrix_diagonal = tl.sum(
            tl.sum(tl.where((row == index) & (column == index), matrix, 0.0), axis=0),
            axis=0,
        )
        diagonal = tl.sqrt(tl.maximum(matrix_diagonal - tl.sum(lower_row * lower_row), epsilon))
        lower = tl.where((row == index) & (column == index), diagonal, lower)
        matrix_column = tl.sum(tl.where(column == index, matrix, 0.0), axis=1)
        projection = tl.sum(lower * lower_row[None, :], axis=1)
        values = (matrix_column - projection) / diagonal
        lower = tl.where(
            (column == index) & (row > index) & (row < N),
            values[:, None],
            lower,
        )

    rhs_column = tl.arange(0, BLOCK_R)[None, :]
    rhs_offset = system_id * N * R + row * R + rhs_column
    right = tl.load(
        rhs_ptr + rhs_offset,
        mask=(system_id < systems) & (row < N) & (rhs_column < R),
        other=0.0,
    ).to(tl.float32)
    forward = tl.zeros((BLOCK_N, BLOCK_R), tl.float32)
    for index in tl.static_range(0, N):
        lower_row = tl.sum(tl.where(row == index, lower, 0.0), axis=0)
        right_row = tl.sum(tl.where(row == index, right, 0.0), axis=0)
        diagonal = tl.sum(tl.where(tl.arange(0, BLOCK_N) == index, lower_row, 0.0), axis=0)
        residual = right_row - tl.sum(lower_row[:, None] * forward, axis=0)
        value = residual / diagonal
        forward = tl.where(row == index, value[None, :], forward)

    solution = tl.zeros((BLOCK_N, BLOCK_R), tl.float32)
    for reverse_index in tl.static_range(0, N):
        lower_column = tl.sum(
            tl.where(column == N - 1 - reverse_index, lower, 0.0), axis=1
        )
        forward_row = tl.sum(
            tl.where(row == N - 1 - reverse_index, forward, 0.0), axis=0
        )
        diagonal = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_N) == N - 1 - reverse_index,
                lower_column,
                0.0,
            ),
            axis=0,
        )
        residual = forward_row - tl.sum(lower_column[:, None] * solution, axis=0)
        value = residual / diagonal
        solution = tl.where(row == N - 1 - reverse_index, value[None, :], solution)
    tl.store(
        output_ptr + rhs_offset,
        solution,
        mask=(system_id < systems) & (row < N) & (rhs_column < R),
    )


@triton_op("joint_mpc_rti::fixed_spd_solve_cuda", mutates_args={})
def fixed_spd_solve_cuda(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(rhs)
    dimension = int(matrix.shape[-1])
    rhs_columns = int(rhs.shape[-1])
    systems = int(matrix.numel()) // (dimension * dimension)
    block_n = triton.next_power_of_2(dimension)
    block_r = triton.next_power_of_2(rhs_columns)
    wrap_triton(_fixed_spd_solve_kernel)[(systems,)](
        matrix,
        rhs,
        output,
        systems,
        N=dimension,
        R=rhs_columns,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
    )
    return output


__all__ = ["fixed_spd_solve_cuda"]
