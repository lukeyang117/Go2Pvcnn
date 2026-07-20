"""Block-banded direct-state trajectory QP contracts and dense references."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


JOINT_LOWER = (-1.0472, -0.6632, -2.721) * 4
JOINT_UPPER = (1.0472, 2.966, -0.837) * 4


@dataclass(frozen=True)
class TrajectoryQp:
    diagonal: Tensor
    first_offdiag: Tensor
    second_offdiag: Tensor
    gradient: Tensor
    lower: Tensor
    upper: Tensor
    joint_difference_lower: Tensor
    joint_difference_upper: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.gradient.shape[0])

    @property
    def nodes(self) -> int:
        return int(self.gradient.shape[1])

    def to_dense(self) -> tuple[Tensor, Tensor]:
        """Materialize the test-only dense Hessian and flattened gradient."""
        batch, nodes, state_dim = self.gradient.shape
        blocks = self.gradient.new_zeros(batch, nodes, nodes, state_dim, state_dim)
        node = torch.arange(nodes, device=self.gradient.device)
        edge = torch.arange(nodes - 1, device=self.gradient.device)
        second = torch.arange(nodes - 2, device=self.gradient.device)
        blocks[:, node, node] = self.diagonal
        blocks[:, edge, edge + 1] = self.first_offdiag
        blocks[:, edge + 1, edge] = self.first_offdiag.transpose(-1, -2)
        blocks[:, second, second + 2] = self.second_offdiag
        blocks[:, second + 2, second] = self.second_offdiag.transpose(-1, -2)
        dense = blocks.permute(0, 1, 3, 2, 4).reshape(batch, nodes * state_dim, nodes * state_dim)
        return dense, self.gradient.flatten(1)


def trajectory_bounds(nominal: Tensor, cfg) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    state = torch.as_tensor(nominal)
    trust = state.new_tensor(
        (cfg.solver.root_position_trust,) * 3
        + (cfg.solver.root_orientation_trust,) * 3
        + (cfg.solver.joint_trust,) * 12
    ).view(1, 1, 18)
    lower = -trust.expand_as(state).clone()
    upper = trust.expand_as(state).clone()
    joint_lower = state.new_tensor(JOINT_LOWER).view(1, 1, 12) - state[..., 6:]
    joint_upper = state.new_tensor(JOINT_UPPER).view(1, 1, 12) - state[..., 6:]
    lower[..., 6:] = torch.maximum(lower[..., 6:], joint_lower)
    upper[..., 6:] = torch.minimum(upper[..., 6:], joint_upper)
    lower[:, 0] = 0.0
    upper[:, 0] = 0.0

    nominal_difference = state[:, 1:, 6:] - state[:, :-1, 6:]
    maximum_step = float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    difference_lower = -maximum_step - nominal_difference
    difference_upper = maximum_step - nominal_difference
    return lower, upper, difference_lower, difference_upper


__all__ = ["JOINT_LOWER", "JOINT_UPPER", "TrajectoryQp", "trajectory_bounds"]
