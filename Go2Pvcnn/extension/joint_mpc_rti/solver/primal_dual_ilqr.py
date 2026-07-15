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
        control_diagonal = control_hessian.diagonal(dim1=-2, dim2=-1).clamp_min(float(regularization))
        solve_state = control_state / control_diagonal.unsqueeze(-1)
        solve_vector = control_vector / control_diagonal
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


def _solve_go2_block_tensors(
    matrix_a: Tensor,
    matrix_b: Tensor,
    matrix_q: Tensor,
    matrix_r: Tensor,
    vector_q: Tensor,
    vector_r: Tensor,
    terminal_q: Tensor,
    terminal_vector: Tensor,
    initial_state: Tensor,
    affine: Tensor,
    matrix_s: Tensor,
    regularization: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    horizon = int(matrix_a.shape[1])
    root_a = matrix_a[..., :6, :6]
    root_b = matrix_b[..., :6, :6]
    root_q = matrix_q[..., :6, :6]
    root_r = matrix_r[..., :6, :6]
    root_s = matrix_s[..., :6, :6]
    root_vector_q = vector_q[..., :6]
    root_vector_r = vector_r[..., :6]
    root_affine = affine[..., :6]
    joint_a = matrix_a.diagonal(dim1=-2, dim2=-1)[..., 6:]
    joint_b = matrix_b.diagonal(dim1=-2, dim2=-1)[..., 6:]
    joint_q = matrix_q.diagonal(dim1=-2, dim2=-1)[..., 6:]
    joint_r = matrix_r.diagonal(dim1=-2, dim2=-1)[..., 6:]
    joint_s = matrix_s.diagonal(dim1=-2, dim2=-1)[..., 6:]
    joint_vector_q = vector_q[..., 6:]
    joint_vector_r = vector_r[..., 6:]
    joint_affine = affine[..., 6:]
    root_value_matrix = terminal_q[..., :6, :6]
    root_value_vector = terminal_vector[..., :6]
    joint_value_matrix = terminal_q.diagonal(dim1=-2, dim2=-1)[..., 6:]
    joint_value_vector = terminal_vector[..., 6:]
    root_feedback_reverse: list[Tensor] = []
    root_feedforward_reverse: list[Tensor] = []
    root_value_matrix_reverse: list[Tensor] = []
    root_value_vector_reverse: list[Tensor] = []
    joint_feedback_reverse: list[Tensor] = []
    joint_feedforward_reverse: list[Tensor] = []
    joint_value_matrix_reverse: list[Tensor] = []
    joint_value_vector_reverse: list[Tensor] = []

    for index in range(horizon - 1, -1, -1):
        stage_a = root_a[:, index]
        stage_b = root_b[:, index]
        stage_a_t = stage_a.transpose(-1, -2)
        stage_b_t = stage_b.transpose(-1, -2)
        value_affine = torch.matmul(root_value_matrix, root_affine[:, index].unsqueeze(-1)).squeeze(-1)
        value_affine = value_affine + root_value_vector
        control_hessian = root_r[:, index] + torch.matmul(
            stage_b_t,
            torch.matmul(root_value_matrix, stage_b),
        ) + float(regularization) * torch.eye(6, dtype=matrix_a.dtype, device=matrix_a.device)
        control_diagonal = control_hessian.diagonal(dim1=-2, dim2=-1).clamp_min(float(regularization))
        control_state = root_s[:, index] + torch.matmul(
            stage_b_t,
            torch.matmul(root_value_matrix, stage_a),
        )
        control_vector = root_vector_r[:, index] + torch.matmul(
            stage_b_t,
            value_affine.unsqueeze(-1),
        ).squeeze(-1)
        solve_state = control_state / control_diagonal.unsqueeze(-1)
        solve_vector = control_vector / control_diagonal
        root_feedback_reverse.append(-solve_state)
        root_feedforward_reverse.append(-solve_vector)
        root_value_matrix = root_q[:, index] + torch.matmul(
            stage_a_t,
            torch.matmul(root_value_matrix, stage_a),
        ) - torch.matmul(control_state.transpose(-1, -2), solve_state)
        root_value_matrix = 0.5 * (root_value_matrix + root_value_matrix.transpose(-1, -2))
        root_value_vector = root_vector_q[:, index] + torch.matmul(
            stage_a_t,
            value_affine.unsqueeze(-1),
        ).squeeze(-1) - torch.matmul(
            control_state.transpose(-1, -2),
            solve_vector.unsqueeze(-1),
        ).squeeze(-1)
        root_value_matrix_reverse.append(root_value_matrix)
        root_value_vector_reverse.append(root_value_vector)

        joint_value_affine = joint_value_matrix * joint_affine[:, index] + joint_value_vector
        joint_control_hessian = (
            joint_r[:, index]
            + joint_b[:, index].square() * joint_value_matrix
            + float(regularization)
        ).clamp_min(float(regularization))
        joint_control_state = joint_s[:, index] + joint_b[:, index] * joint_value_matrix * joint_a[:, index]
        joint_control_vector = joint_vector_r[:, index] + joint_b[:, index] * joint_value_affine
        joint_solve_state = joint_control_state / joint_control_hessian
        joint_solve_vector = joint_control_vector / joint_control_hessian
        joint_feedback_reverse.append(-joint_solve_state)
        joint_feedforward_reverse.append(-joint_solve_vector)
        joint_value_matrix = (
            joint_q[:, index]
            + joint_a[:, index].square() * joint_value_matrix
            - joint_control_state * joint_solve_state
        )
        joint_value_vector = (
            joint_vector_q[:, index]
            + joint_a[:, index] * joint_value_affine
            - joint_control_state * joint_solve_vector
        )
        joint_value_matrix_reverse.append(joint_value_matrix)
        joint_value_vector_reverse.append(joint_value_vector)

    root_feedback = torch.stack(root_feedback_reverse[::-1], dim=1)
    root_feedforward = torch.stack(root_feedforward_reverse[::-1], dim=1)
    root_values = torch.stack(root_value_matrix_reverse[::-1], dim=1)
    root_vectors = torch.stack(root_value_vector_reverse[::-1], dim=1)
    joint_feedback = torch.stack(joint_feedback_reverse[::-1], dim=1)
    joint_feedforward = torch.stack(joint_feedforward_reverse[::-1], dim=1)
    joint_values = torch.stack(joint_value_matrix_reverse[::-1], dim=1)
    joint_vectors = torch.stack(joint_value_vector_reverse[::-1], dim=1)
    root_state = initial_state[..., :6]
    joint_state = initial_state[..., 6:]
    root_states = [root_state]
    joint_states = [joint_state]
    root_controls: list[Tensor] = []
    joint_controls: list[Tensor] = []
    for index in range(horizon):
        root_control = torch.matmul(root_feedback[:, index], root_state.unsqueeze(-1)).squeeze(-1)
        root_control = root_control + root_feedforward[:, index]
        joint_control = joint_feedback[:, index] * joint_state + joint_feedforward[:, index]
        root_state = (
            torch.matmul(root_a[:, index], root_state.unsqueeze(-1)).squeeze(-1)
            + torch.matmul(root_b[:, index], root_control.unsqueeze(-1)).squeeze(-1)
            + root_affine[:, index]
        )
        joint_state = joint_a[:, index] * joint_state + joint_b[:, index] * joint_control + joint_affine[:, index]
        root_controls.append(root_control)
        joint_controls.append(joint_control)
        root_states.append(root_state)
        joint_states.append(joint_state)
    state = torch.cat((torch.stack(root_states, dim=1), torch.stack(joint_states, dim=1)), dim=-1)
    control = torch.cat((torch.stack(root_controls, dim=1), torch.stack(joint_controls, dim=1)), dim=-1)
    dual = torch.cat(
        (
            torch.matmul(root_values, state[:, :-1, :6].unsqueeze(-1)).squeeze(-1) + root_vectors,
            joint_values * state[:, :-1, 6:] + joint_vectors,
        ),
        dim=-1,
    )
    feedback = torch.zeros_like(matrix_a)
    feedback[..., :6, :6] = root_feedback
    feedback[..., 6:, 6:] = torch.diag_embed(joint_feedback)
    feedforward = torch.cat((root_feedforward, joint_feedforward), dim=-1)
    return state, control, feedback, feedforward, dual


_COMPILED_GO2_BLOCK_SOLVE = torch.compile(
    _solve_go2_block_tensors,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def solve_go2_block_lq_subproblem(problem: LqProblem, *, regularization: float) -> LqSolution:
    matrix_a = torch.as_tensor(problem.matrix_a)
    matrix_s = (
        torch.zeros_like(matrix_a)
        if problem.matrix_s is None
        else torch.as_tensor(problem.matrix_s, dtype=matrix_a.dtype, device=matrix_a.device)
    )
    arguments = (
        matrix_a,
        torch.as_tensor(problem.matrix_b, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.matrix_q, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.matrix_r, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.vector_q, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.vector_r, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.terminal_q, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.terminal_vector, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.initial_state, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.affine_dynamics, dtype=matrix_a.dtype, device=matrix_a.device),
        matrix_s,
        float(regularization),
    )
    outputs = _COMPILED_GO2_BLOCK_SOLVE(*arguments) if matrix_a.is_cuda else _solve_go2_block_tensors(*arguments)
    return LqSolution(
        delta_state=outputs[0],
        delta_control=outputs[1],
        feedback_gain=outputs[2],
        feedforward=outputs[3],
        dual=outputs[4],
    )


def _solve_diagonal_lq_tensors(
    matrix_a: Tensor,
    matrix_b: Tensor,
    matrix_q: Tensor,
    matrix_r: Tensor,
    vector_q: Tensor,
    vector_r: Tensor,
    terminal_q: Tensor,
    terminal_vector: Tensor,
    initial_state: Tensor,
    affine: Tensor,
    matrix_s: Tensor,
    regularization: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    horizon = int(matrix_a.shape[1])
    diagonal_a = matrix_a.diagonal(dim1=-2, dim2=-1)
    diagonal_b = matrix_b.diagonal(dim1=-2, dim2=-1)
    diagonal_q = matrix_q.diagonal(dim1=-2, dim2=-1)
    diagonal_r = matrix_r.diagonal(dim1=-2, dim2=-1)
    diagonal_s = matrix_s.diagonal(dim1=-2, dim2=-1)
    value_matrix = terminal_q.diagonal(dim1=-2, dim2=-1)
    value_vector = terminal_vector
    feedback_reverse: list[Tensor] = []
    feedforward_reverse: list[Tensor] = []
    value_matrix_reverse: list[Tensor] = []
    value_vector_reverse: list[Tensor] = []
    for index in range(horizon - 1, -1, -1):
        value_affine = value_matrix * affine[:, index] + value_vector
        control_hessian = (
            diagonal_r[:, index]
            + diagonal_b[:, index].square() * value_matrix
            + float(regularization)
        ).clamp_min(float(regularization))
        control_state = diagonal_s[:, index] + diagonal_b[:, index] * value_matrix * diagonal_a[:, index]
        control_vector = vector_r[:, index] + diagonal_b[:, index] * value_affine
        solve_state = control_state / control_hessian
        solve_vector = control_vector / control_hessian
        feedback_reverse.append(-solve_state)
        feedforward_reverse.append(-solve_vector)
        value_matrix = (
            diagonal_q[:, index]
            + diagonal_a[:, index].square() * value_matrix
            - control_state * solve_state
        )
        value_vector = (
            vector_q[:, index]
            + diagonal_a[:, index] * value_affine
            - control_state * solve_vector
        )
        value_matrix_reverse.append(value_matrix)
        value_vector_reverse.append(value_vector)
    feedback_diagonal = torch.stack(feedback_reverse[::-1], dim=1)
    feedforward = torch.stack(feedforward_reverse[::-1], dim=1)
    values = torch.stack(value_matrix_reverse[::-1], dim=1)
    vectors = torch.stack(value_vector_reverse[::-1], dim=1)
    state = initial_state
    states = [state]
    controls: list[Tensor] = []
    for index in range(horizon):
        control = feedback_diagonal[:, index] * state + feedforward[:, index]
        state = diagonal_a[:, index] * state + diagonal_b[:, index] * control + affine[:, index]
        controls.append(control)
        states.append(state)
    state_sequence = torch.stack(states, dim=1)
    control_sequence = torch.stack(controls, dim=1)
    dual = values * state_sequence[:, :-1] + vectors
    return state_sequence, control_sequence, torch.diag_embed(feedback_diagonal), feedforward, dual


_COMPILED_DIAGONAL_LQ_SOLVE = torch.compile(
    _solve_diagonal_lq_tensors,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def solve_diagonal_lq_subproblem(problem: LqProblem, *, regularization: float) -> LqSolution:
    matrix_a = torch.as_tensor(problem.matrix_a)
    matrix_s = (
        torch.zeros_like(matrix_a)
        if problem.matrix_s is None
        else torch.as_tensor(problem.matrix_s, dtype=matrix_a.dtype, device=matrix_a.device)
    )
    arguments = (
        matrix_a,
        torch.as_tensor(problem.matrix_b, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.matrix_q, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.matrix_r, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.vector_q, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.vector_r, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.terminal_q, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.terminal_vector, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.initial_state, dtype=matrix_a.dtype, device=matrix_a.device),
        torch.as_tensor(problem.affine_dynamics, dtype=matrix_a.dtype, device=matrix_a.device),
        matrix_s,
        float(regularization),
    )
    outputs = _COMPILED_DIAGONAL_LQ_SOLVE(*arguments) if matrix_a.is_cuda else _solve_diagonal_lq_tensors(*arguments)
    return LqSolution(
        delta_state=outputs[0],
        delta_control=outputs[1],
        feedback_gain=outputs[2],
        feedforward=outputs[3],
        dual=outputs[4],
    )


__all__ = [
    "LqProblem",
    "LqSolution",
    "solve_diagonal_lq_subproblem",
    "solve_go2_block_lq_subproblem",
    "solve_lq_subproblem",
]
