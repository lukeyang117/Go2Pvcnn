"""Optimization variables and decode helpers for batch MPC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import MpcRuntimeCfg


@dataclass
class MpcOptimizationVariables:
    root_pos_residual: Tensor
    root_rpy_residual: Tensor
    foot_pos_residual: Tensor
    contact_logits: Tensor

    def parameters(self) -> list[Tensor]:
        return [self.root_pos_residual, self.root_rpy_residual, self.foot_pos_residual, self.contact_logits]


@dataclass(frozen=True)
class DecodedMpcTrajectory:
    root_pos: Tensor
    root_rpy: Tensor
    foot_pos: Tensor
    contact_logits: Tensor
    contact_prob: Tensor


def _clone_or_zeros(value: Tensor | None, like: Tensor) -> Tensor:
    if value is None:
        return torch.zeros_like(like)
    out = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tuple(out.shape) != tuple(like.shape):
        return torch.zeros_like(like)
    return out.clone()


def init_optimization_variables(
    nominal: dict[str, Tensor],
    runtime_cfg: MpcRuntimeCfg,
    *,
    warm_start: MpcOptimizationVariables | None = None,
) -> MpcOptimizationVariables:
    # Rollout can invoke planner under torch.inference_mode(); create optimizer
    # variables in normal autograd mode so Adam can update them in-place.
    with torch.inference_mode(False):
        root_pos_like = nominal["root_pos"]
        root_rpy_like = nominal["root_rpy"]
        foot_pos_like = nominal["foot_pos"]
        contact_like = nominal["contact_logits"]
        if warm_start is not None and runtime_cfg.warm_start_from_previous_plan:
            root_pos_residual = _clone_or_zeros(warm_start.root_pos_residual, root_pos_like)
            root_rpy_residual = _clone_or_zeros(warm_start.root_rpy_residual, root_rpy_like)
            foot_pos_residual = _clone_or_zeros(warm_start.foot_pos_residual, foot_pos_like)
            contact_logits = _clone_or_zeros(warm_start.contact_logits, contact_like)
        else:
            root_pos_residual = torch.zeros_like(root_pos_like)
            root_rpy_residual = torch.zeros_like(root_rpy_like)
            foot_pos_residual = torch.zeros_like(foot_pos_like)
            contact_logits = contact_like.clone()

        root_pos_residual = root_pos_residual.detach().clone().requires_grad_(True)
        root_rpy_residual = root_rpy_residual.detach().clone().requires_grad_(True)
        foot_pos_residual = foot_pos_residual.detach().clone().requires_grad_(True)
        contact_logits = contact_logits.detach().clone().requires_grad_(True)
        return MpcOptimizationVariables(
            root_pos_residual=root_pos_residual,
            root_rpy_residual=root_rpy_residual,
            foot_pos_residual=foot_pos_residual,
            contact_logits=contact_logits,
        )


def decode_trajectory(
    nominal: dict[str, Tensor],
    variables: MpcOptimizationVariables,
    runtime_cfg: MpcRuntimeCfg,
) -> DecodedMpcTrajectory:
    root_pos = nominal["root_pos"] + variables.root_pos_residual
    root_rpy = nominal["root_rpy"] + variables.root_rpy_residual
    foot_pos = nominal["foot_pos"] + variables.foot_pos_residual
    contact_logits = variables.contact_logits
    contact_prob = torch.sigmoid(contact_logits / float(runtime_cfg.contact_temperature))
    return DecodedMpcTrajectory(
        root_pos=root_pos,
        root_rpy=root_rpy,
        foot_pos=foot_pos,
        contact_logits=contact_logits,
        contact_prob=contact_prob,
    )


__all__ = [
    "DecodedMpcTrajectory",
    "MpcOptimizationVariables",
    "decode_trajectory",
    "init_optimization_variables",
]
