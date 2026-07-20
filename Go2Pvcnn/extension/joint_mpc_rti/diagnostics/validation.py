"""Tensor-only result validation reduced after synchronized performance runs."""

from __future__ import annotations

import torch

from extension.joint_mpc_rti.types import JointMpcRtiStepResult


def nonfinite_count(result: JointMpcRtiStepResult) -> int:
    trajectory = result.full_trajectory
    count = (
        torch.logical_not(torch.isfinite(trajectory.state)).sum()
        + torch.logical_not(torch.isfinite(trajectory.derived_velocity)).sum()
        + torch.logical_not(torch.isfinite(trajectory.foot_pos_w)).sum()
    )
    return int(count.item())


__all__ = ["nonfinite_count"]
