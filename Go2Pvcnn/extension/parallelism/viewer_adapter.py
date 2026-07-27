from __future__ import annotations

from types import SimpleNamespace

import torch

from extension.parallelism.types import ParallelismTrajectory


def _yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    return torch.stack(
        (
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        ),
        dim=-1,
    )


def parallelism_trajectory_to_viewer_result(trajectory: ParallelismTrajectory):
    root_quat_w = _yaw_to_quat_wxyz(trajectory.root_rpy_w[..., 2])
    return SimpleNamespace(
        num_frames=int(trajectory.root_pos_w.shape[1]),
        root_pos_w=trajectory.root_pos_w,
        root_quat_w=root_quat_w,
        joint_angles=trajectory.joint_pos,
        foot_pos_w=trajectory.foot_pos_w,
        foot_pos_root=trajectory.foot_pos_w - trajectory.root_pos_w.unsqueeze(2),
        contact_state=trajectory.contact_state,
        planned_touchdown_w=trajectory.selected_foothold_w,
        feasible=trajectory.valid,
        status=(~trajectory.valid).to(torch.long),
        safe_fallback=torch.zeros_like(trajectory.valid),
        parallelism_diagnostics=trajectory.diagnostics,
        joint_mpc_diagnostics=None,
        nominal_state=None,
        alpha_candidate_state=None,
        gait_phase=None,
        publish=None,
        stop=None,
        loss_breakdown=None,
    )
