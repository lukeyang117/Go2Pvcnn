"""Smoothness losses for root and feet."""

from __future__ import annotations

import torch
from torch import Tensor


def root_smoothness_loss(root_pos: Tensor, root_rpy: Tensor) -> Tensor:
    if int(root_pos.shape[1]) < 2:
        return torch.zeros(root_pos.shape[0], dtype=root_pos.dtype, device=root_pos.device)
    dpos = root_pos[:, 1:] - root_pos[:, :-1]
    drpy = root_rpy[:, 1:] - root_rpy[:, :-1]
    return torch.linalg.norm(dpos, dim=-1).mean(dim=-1) + torch.linalg.norm(drpy, dim=-1).mean(dim=-1)


def foot_smoothness_loss(foot_pos: Tensor) -> Tensor:
    if int(foot_pos.shape[1]) < 2:
        return torch.zeros(foot_pos.shape[0], dtype=foot_pos.dtype, device=foot_pos.device)
    dfoot = foot_pos[:, 1:] - foot_pos[:, :-1]
    return torch.linalg.norm(dfoot, dim=-1).mean(dim=(1, 2))


__all__ = ["foot_smoothness_loss", "root_smoothness_loss"]
