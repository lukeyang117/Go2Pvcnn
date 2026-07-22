"""Single-kernel fixed-size general solves for CUDA graph workloads."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _fixed_general_solve_kernel(
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
    row_index = tl.arange(0, BLOCK_N)
    row = row_index[:, None]
    column = tl.arange(0, BLOCK_N)[None, :]
    matrix_offset = system_id * N * N + row * N + column
    matrix = tl.load(
        matrix_ptr + matrix_offset,
        mask=(system_id < systems) & (row < N) & (column < N),
        other=0.0,
    ).to(tl.float32)
    rhs_column = tl.arange(0, BLOCK_R)[None, :]
    rhs_offset = system_id * N * R + row * R + rhs_column
    right = tl.load(
        rhs_ptr + rhs_offset,
        mask=(system_id < systems) & (row < N) & (rhs_column < R),
        other=0.0,
    ).to(tl.float32)

    for pivot_index in tl.static_range(0, N):
        matrix_column = tl.sum(tl.where(column == pivot_index, matrix, 0.0), axis=1)
        pivot_row_index = tl.argmax(
            tl.where(
                (row_index >= pivot_index) & (row_index < N),
                tl.abs(matrix_column),
                -1.0,
            ),
            axis=0,
        )
        pivot_row = tl.sum(tl.where(row == pivot_index, matrix, 0.0), axis=0)
        pivot_rhs = tl.sum(tl.where(row == pivot_index, right, 0.0), axis=0)
        selected_row = tl.sum(tl.where(row == pivot_row_index, matrix, 0.0), axis=0)
        selected_rhs = tl.sum(tl.where(row == pivot_row_index, right, 0.0), axis=0)
        matrix = tl.where(
            row == pivot_index,
            selected_row[None, :],
            tl.where(row == pivot_row_index, pivot_row[None, :], matrix),
        )
        right = tl.where(
            row == pivot_index,
            selected_rhs[None, :],
            tl.where(row == pivot_row_index, pivot_rhs[None, :], right),
        )
        pivot_row = tl.sum(tl.where(row == pivot_index, matrix, 0.0), axis=0)
        pivot_rhs = tl.sum(tl.where(row == pivot_index, right, 0.0), axis=0)
        pivot = tl.sum(
            tl.where(tl.arange(0, BLOCK_N) == pivot_index, pivot_row, 0.0), axis=0
        )
        matrix_column = tl.sum(tl.where(column == pivot_index, matrix, 0.0), axis=1)
        factor = tl.where(row_index > pivot_index, matrix_column / pivot, 0.0)
        matrix = tl.where(
            row > pivot_index,
            matrix - factor[:, None] * pivot_row[None, :],
            matrix,
        )
        right = tl.where(
            row > pivot_index,
            right - factor[:, None] * pivot_rhs[None, :],
            right,
        )

    solution = tl.zeros((BLOCK_N, BLOCK_R), tl.float32)
    for reverse_index in tl.static_range(0, N):
        upper_row = tl.sum(
            tl.where(row == N - 1 - reverse_index, matrix, 0.0), axis=0
        )
        rhs_row = tl.sum(
            tl.where(row == N - 1 - reverse_index, right, 0.0), axis=0
        )
        diagonal = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_N) == N - 1 - reverse_index,
                upper_row,
                0.0,
            ),
            axis=0,
        )
        residual = rhs_row - tl.sum(upper_row[:, None] * solution, axis=0)
        solution = tl.where(
            row == N - 1 - reverse_index,
            (residual / diagonal)[None, :],
            solution,
        )
    tl.store(
        output_ptr + rhs_offset,
        solution,
        mask=(system_id < systems) & (row < N) & (rhs_column < R),
    )


@triton_op("joint_mpc_rti::fixed_general_solve_cuda", mutates_args={})
def fixed_general_solve_cuda(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(rhs)
    dimension = int(matrix.shape[-1])
    rhs_columns = int(rhs.shape[-1])
    systems = int(matrix.numel()) // (dimension * dimension)
    wrap_triton(_fixed_general_solve_kernel)[(systems,)](
        matrix,
        rhs,
        output,
        systems,
        N=dimension,
        R=rhs_columns,
        BLOCK_N=triton.next_power_of_2(dimension),
        BLOCK_R=triton.next_power_of_2(rhs_columns),
    )
    return output


__all__ = ["fixed_general_solve_cuda"]
