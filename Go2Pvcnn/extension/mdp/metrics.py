"""Tracking metrics for planner-guided trajectory imitation."""

from __future__ import annotations

import torch


def compute_tracking_metrics(
    *,
    root_pos: torch.Tensor,
    ref_root_pos: torch.Tensor,
    root_yaw: torch.Tensor,
    ref_root_yaw: torch.Tensor,
    joint_pos: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    foot_pos_root: torch.Tensor,
    ref_foot_pos_root: torch.Tensor,
    contact_state: torch.Tensor,
    ref_contact_state: torch.Tensor,
    touchdown_pos: torch.Tensor,
    ref_touchdown_pos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute scalar tracking metrics from current and reference states."""
    root_xy_error = torch.linalg.norm(root_pos[:, :2] - ref_root_pos[:, :2], dim=1).mean()
    root_z_error = torch.abs(root_pos[:, 2] - ref_root_pos[:, 2]).mean()
    root_yaw_error = torch.abs(root_yaw - ref_root_yaw).mean()
    joint_error = torch.abs(joint_pos - ref_joint_pos).mean()
    foot_error = torch.linalg.norm(foot_pos_root - ref_foot_pos_root, dim=-1).mean()
    contact_match_rate = (contact_state == ref_contact_state).float().mean()
    touchdown_error = torch.linalg.norm(touchdown_pos - ref_touchdown_pos, dim=-1).mean()
    tracking_score = (
        root_xy_error
        + root_z_error
        + root_yaw_error
        + joint_error
        + foot_error
        + touchdown_error
        + (1.0 - contact_match_rate)
    )

    return {
        "root_xy_error_mean": root_xy_error,
        "root_z_error_mean": root_z_error,
        "root_yaw_error_mean": root_yaw_error,
        "joint_error_mean": joint_error,
        "foot_pos_root_error_mean": foot_error,
        "contact_match_rate": contact_match_rate,
        "touchdown_error_mean": touchdown_error,
        "trajectory_tracking_score": tracking_score,
    }
