"""Hard-diagnostics helpers for batch MPC results."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import MPC_HARD_REASON_NAMES, MpcPlannerStatus


def evaluate_hard_reasons(
    *,
    root_pos: Tensor,
    foot_pos: Tensor,
    joint_angles: Tensor,
    contact_state: Tensor,
    command: Tensor,
) -> Tensor:
    """Return hard reason mask shaped [B, R]."""
    foot_min_z = foot_pos[..., 2].amin(dim=(1, 2))
    large_collision = foot_min_z < -0.20
    small_collision = foot_min_z < -0.05
    touchdown_unsupported = contact_state.to(dtype=torch.float32).sum(dim=(1, 2)) < 1.0
    ik_workspace = torch.abs(joint_angles).amax(dim=(1, 2)) > 2.8
    airborne_unstable = (contact_state.sum(dim=-1) == 0).any(dim=1)
    progress = root_pos[:, -1, 0] - root_pos[:, 0, 0]
    cmd_forward = command[:, 0]
    cmd_progress_violation = torch.logical_and(cmd_forward > 0.05, progress < 0.01)
    return torch.stack(
        (
            large_collision,
            small_collision,
            touchdown_unsupported,
            ik_workspace,
            airborne_unstable,
            cmd_progress_violation,
        ),
        dim=-1,
    )


def status_from_hard_reasons(hard_reason_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    any_bad = hard_reason_mask.any(dim=-1)
    feasible = torch.logical_not(any_bad)
    safe_fallback = torch.logical_not(feasible)
    status = torch.where(
        feasible,
        torch.full_like(any_bad, int(MpcPlannerStatus.OK), dtype=torch.long),
        torch.full_like(any_bad, int(MpcPlannerStatus.ALL_INFEASIBLE), dtype=torch.long),
    )
    return status, feasible, safe_fallback


__all__ = ["MPC_HARD_REASON_NAMES", "evaluate_hard_reasons", "status_from_hard_reasons"]
