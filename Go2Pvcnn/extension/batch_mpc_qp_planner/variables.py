"""Fixed QP variable layout for the MPC-QP backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class QpVariableLayout:
    batch: int
    horizon: int
    legs: int
    total_dim: int
    root_xy: Tensor
    foot_control_xyz: Tensor
    touchdown_xy: Tensor
    semantic_slack: Tensor
    clearance_slack: Tensor
    reachability_slack: Tensor
    stability_slack: Tensor
    touchdown_xy_indices: Tensor
    touchdown_z_is_height_bound: bool = True


def _take(cursor: int, count: int, *, batch: int, device: torch.device) -> tuple[Tensor, int]:
    idx = torch.arange(cursor, cursor + int(count), dtype=torch.long, device=device)
    return idx.unsqueeze(0).expand(int(batch), -1).contiguous(), cursor + int(count)


def build_qp_variable_layout(
    *,
    batch: int,
    horizon: int,
    legs: int = 4,
    device: torch.device,
) -> QpVariableLayout:
    cursor = 0
    root_flat, cursor = _take(cursor, int(horizon) * 2, batch=batch, device=device)
    foot_flat, cursor = _take(cursor, int(legs) * 2 * 3, batch=batch, device=device)
    touchdown_flat, cursor = _take(cursor, int(legs) * 2, batch=batch, device=device)
    semantic_flat, cursor = _take(cursor, int(legs), batch=batch, device=device)
    clearance_flat, cursor = _take(cursor, int(legs), batch=batch, device=device)
    reachability_flat, cursor = _take(cursor, int(legs), batch=batch, device=device)
    stability_flat, cursor = _take(cursor, 1, batch=batch, device=device)
    return QpVariableLayout(
        batch=int(batch),
        horizon=int(horizon),
        legs=int(legs),
        total_dim=cursor,
        root_xy=root_flat.reshape(int(batch), int(horizon), 2),
        foot_control_xyz=foot_flat.reshape(int(batch), int(legs), 2, 3),
        touchdown_xy=touchdown_flat.reshape(int(batch), int(legs), 2),
        semantic_slack=semantic_flat.reshape(int(batch), int(legs)),
        clearance_slack=clearance_flat.reshape(int(batch), int(legs)),
        reachability_slack=reachability_flat.reshape(int(batch), int(legs)),
        stability_slack=stability_flat.reshape(int(batch), 1),
        touchdown_xy_indices=touchdown_flat.reshape(int(batch), int(legs), 2),
    )


__all__ = ["QpVariableLayout", "build_qp_variable_layout"]
