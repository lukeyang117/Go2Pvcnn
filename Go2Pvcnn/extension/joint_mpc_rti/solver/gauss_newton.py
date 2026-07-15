"""Generalized Gauss-Newton block assembly for RTI subproblems."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.solver.primal_dual_ilqr import LqProblem


@dataclass(frozen=True)
class ResidualLinearization:
    residual: Tensor
    jacobian_x: Tensor
    jacobian_u: Tensor
    terminal_residual: Tensor
    terminal_jacobian_x: Tensor


def build_gauss_newton_lq(
    linearization: ResidualLinearization,
    *,
    matrix_a: Tensor,
    matrix_b: Tensor,
    affine_dynamics: Tensor,
    initial_delta_state: Tensor,
    regularization: float,
) -> LqProblem:
    residual = torch.as_tensor(linearization.residual)
    jacobian_x = torch.as_tensor(linearization.jacobian_x, dtype=residual.dtype, device=residual.device)
    jacobian_u = torch.as_tensor(linearization.jacobian_u, dtype=residual.dtype, device=residual.device)
    terminal_residual = torch.as_tensor(
        linearization.terminal_residual, dtype=residual.dtype, device=residual.device
    )
    terminal_jacobian = torch.as_tensor(
        linearization.terminal_jacobian_x, dtype=residual.dtype, device=residual.device
    )
    jacobian_x_t = jacobian_x.transpose(-1, -2)
    jacobian_u_t = jacobian_u.transpose(-1, -2)
    matrix_q = torch.matmul(jacobian_x_t, jacobian_x)
    matrix_r = torch.matmul(jacobian_u_t, jacobian_u)
    control_dim = int(matrix_r.shape[-1])
    matrix_r = matrix_r + float(regularization) * torch.eye(
        control_dim, dtype=residual.dtype, device=residual.device
    )
    matrix_s = torch.matmul(jacobian_u_t, jacobian_x)
    vector_q = torch.matmul(jacobian_x_t, residual.unsqueeze(-1)).squeeze(-1)
    vector_r = torch.matmul(jacobian_u_t, residual.unsqueeze(-1)).squeeze(-1)
    terminal_jacobian_t = terminal_jacobian.transpose(-1, -2)
    terminal_q = torch.matmul(terminal_jacobian_t, terminal_jacobian)
    terminal_vector = torch.matmul(terminal_jacobian_t, terminal_residual.unsqueeze(-1)).squeeze(-1)
    return LqProblem(
        matrix_a=torch.as_tensor(matrix_a, dtype=residual.dtype, device=residual.device),
        matrix_b=torch.as_tensor(matrix_b, dtype=residual.dtype, device=residual.device),
        matrix_q=matrix_q,
        matrix_r=matrix_r,
        vector_q=vector_q,
        vector_r=vector_r,
        terminal_q=terminal_q,
        terminal_vector=terminal_vector,
        initial_state=torch.as_tensor(initial_delta_state, dtype=residual.dtype, device=residual.device),
        affine_dynamics=torch.as_tensor(affine_dynamics, dtype=residual.dtype, device=residual.device),
        matrix_s=matrix_s,
    )


__all__ = ["ResidualLinearization", "build_gauss_newton_lq"]
