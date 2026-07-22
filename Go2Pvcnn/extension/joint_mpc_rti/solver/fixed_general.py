"""Fixed-size general linear solves with a graph-safe CUDA backend."""

from __future__ import annotations

import torch
from torch import Tensor


def fixed_general_solve(matrix: Tensor, rhs: Tensor) -> Tensor:
    system = torch.as_tensor(matrix)
    right = torch.as_tensor(rhs, dtype=system.dtype, device=system.device)
    if system.ndim < 2 or system.shape[-1] != system.shape[-2]:
        raise ValueError("matrix must have shape [..., N, N]")
    if right.shape[:-2] != system.shape[:-2] or right.shape[-2] != system.shape[-1]:
        raise ValueError("rhs must have shape [..., N, R]")
    if system.is_cuda:
        from extension.joint_mpc_rti.solver.fixed_general_triton import fixed_general_solve_cuda

        return fixed_general_solve_cuda(system, right)
    return torch.linalg.solve_ex(system, right, check_errors=False)[0]


__all__ = ["fixed_general_solve"]
