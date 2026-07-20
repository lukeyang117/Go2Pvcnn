"""First- and second-difference state trajectory regularization."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg


def smooth_residual(state: Tensor, cfg: JointMpcRtiCfg) -> Tensor:
    trajectory = torch.as_tensor(state)
    first = math.sqrt(float(cfg.loss_terms.smooth_first)) * (trajectory[:, 1:] - trajectory[:, :-1])
    second = math.sqrt(float(cfg.loss_terms.smooth_second)) * (
        trajectory[:, 2:] - 2.0 * trajectory[:, 1:-1] + trajectory[:, :-2]
    )
    first = first.flatten(1) / math.sqrt(float(first[0].numel()))
    second = second.flatten(1) / math.sqrt(float(second[0].numel()))
    return torch.cat((first, second), dim=1)


def smooth_loss(state: Tensor, cfg: JointMpcRtiCfg) -> Tensor:
    residual = smooth_residual(state, cfg)
    return 0.5 * residual.square().sum(dim=1)


__all__ = ["smooth_loss", "smooth_residual"]
