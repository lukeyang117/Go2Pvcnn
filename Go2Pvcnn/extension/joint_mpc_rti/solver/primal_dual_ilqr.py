"""Batched LQ subproblem solve for one SQP real-time iteration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch._higher_order_ops.scan import scan


@dataclass(frozen=True)
class LqProblem:
    matrix_a: Tensor
    matrix_b: Tensor
    matrix_q: Tensor
    matrix_r: Tensor
    vector_q: Tensor
    vector_r: Tensor
    terminal_q: Tensor
    terminal_vector: Tensor
    initial_state: Tensor
    affine_dynamics: Tensor
    matrix_s: Tensor | None = None


@dataclass(frozen=True)
class LqSolution:
    delta_state: Tensor
    delta_control: Tensor
    feedback_gain: Tensor
    feedforward: Tensor
    dual: Tensor


def solve_lq_subproblem(problem: LqProblem, *, regularization: float) -> LqSolution:
    """Solve a fixed-horizon affine LQ problem with graph-level backward/forward scans."""
    matrix_a = torch.as_tensor(problem.matrix_a)
    matrix_b = torch.as_tensor(problem.matrix_b, dtype=matrix_a.dtype, device=matrix_a.device)
    matrix_q = torch.as_tensor(problem.matrix_q, dtype=matrix_a.dtype, device=matrix_a.device)
    matrix_r = torch.as_tensor(problem.matrix_r, dtype=matrix_a.dtype, device=matrix_a.device)
    vector_q = torch.as_tensor(problem.vector_q, dtype=matrix_a.dtype, device=matrix_a.device)
    vector_r = torch.as_tensor(problem.vector_r, dtype=matrix_a.dtype, device=matrix_a.device)
    terminal_q = torch.as_tensor(problem.terminal_q, dtype=matrix_a.dtype, device=matrix_a.device)
    terminal_vector = torch.as_tensor(problem.terminal_vector, dtype=matrix_a.dtype, device=matrix_a.device)
    initial_state = torch.as_tensor(problem.initial_state, dtype=matrix_a.dtype, device=matrix_a.device)
    affine = torch.as_tensor(problem.affine_dynamics, dtype=matrix_a.dtype, device=matrix_a.device)
    matrix_s = (
        torch.zeros(
            *matrix_a.shape[:2],
            matrix_b.shape[-1],
            matrix_a.shape[-1],
            dtype=matrix_a.dtype,
            device=matrix_a.device,
        )
        if problem.matrix_s is None
        else torch.as_tensor(problem.matrix_s, dtype=matrix_a.dtype, device=matrix_a.device)
    )
    control_dim = int(matrix_b.shape[-1])
    regularizer = float(regularization) * torch.eye(control_dim, dtype=matrix_a.dtype, device=matrix_a.device)

    def backward(
        carry: tuple[Tensor, Tensor],
        stage: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor, Tensor, Tensor]]:
        value_matrix_next, value_vector_next = carry
        stage_a, stage_b, stage_q, stage_r, stage_s, stage_vector_q, stage_vector_r, stage_affine = stage
        b_transpose = stage_b.transpose(-1, -2)
        a_transpose = stage_a.transpose(-1, -2)
        value_affine = torch.matmul(value_matrix_next, stage_affine.unsqueeze(-1)).squeeze(-1) + value_vector_next
        control_hessian = stage_r + torch.matmul(b_transpose, torch.matmul(value_matrix_next, stage_b)) + regularizer
        control_state = stage_s + torch.matmul(b_transpose, torch.matmul(value_matrix_next, stage_a))
        control_vector = stage_vector_r + torch.matmul(b_transpose, value_affine.unsqueeze(-1)).squeeze(-1)
        solve_state = torch.linalg.solve(control_hessian, control_state)
        solve_vector = torch.linalg.solve(control_hessian, control_vector.unsqueeze(-1)).squeeze(-1)
        feedback = -solve_state
        feedforward = -solve_vector
        value_matrix = stage_q + torch.matmul(a_transpose, torch.matmul(value_matrix_next, stage_a)) - torch.matmul(
            control_state.transpose(-1, -2), solve_state
        )
        value_matrix = 0.5 * (value_matrix + value_matrix.transpose(-1, -2))
        value_vector = stage_vector_q + torch.matmul(a_transpose, value_affine.unsqueeze(-1)).squeeze(-1) - torch.matmul(
            control_state.transpose(-1, -2), solve_vector.unsqueeze(-1)
        ).squeeze(-1)
        return (value_matrix, value_vector), (feedback, feedforward, value_matrix, value_vector)

    _, backward_output = scan(
        backward,
        (terminal_q, terminal_vector),
        (matrix_a, matrix_b, matrix_q, matrix_r, matrix_s, vector_q, vector_r, affine),
        dim=1,
        reverse=True,
    )
    feedback, feedforward, value_matrix, value_vector = (
        tensor.movedim(0, 1) for tensor in backward_output
    )

    def forward(
        state: Tensor,
        stage: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        stage_a, stage_b, stage_affine, stage_feedback, stage_feedforward = stage
        control = torch.matmul(stage_feedback, state.unsqueeze(-1)).squeeze(-1) + stage_feedforward
        next_state = (
            torch.matmul(stage_a, state.unsqueeze(-1)).squeeze(-1)
            + torch.matmul(stage_b, control.unsqueeze(-1)).squeeze(-1)
            + stage_affine
        )
        return next_state, (next_state, control)

    _, forward_output = scan(
        forward,
        initial_state,
        (matrix_a, matrix_b, affine, feedback, feedforward),
        dim=1,
    )
    next_state, control = (tensor.movedim(0, 1) for tensor in forward_output)
    state = torch.cat((initial_state.unsqueeze(1), next_state), dim=1)
    dual = torch.matmul(value_matrix, state[:, :-1].unsqueeze(-1)).squeeze(-1) + value_vector
    return LqSolution(
        delta_state=state,
        delta_control=control,
        feedback_gain=feedback,
        feedforward=feedforward,
        dual=dual,
    )


__all__ = ["LqProblem", "LqSolution", "solve_lq_subproblem"]
