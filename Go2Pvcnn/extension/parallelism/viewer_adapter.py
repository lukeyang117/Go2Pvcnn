from __future__ import annotations

from types import SimpleNamespace

import torch

from extension.parallelism.collision import build_official_surface_points_l
from extension.parallelism.config import ParallelismCfg
from extension.parallelism.kinematics import fk_go2
from extension.parallelism.types import ParallelismTrajectory
from extension.convention import euler_to_quat_batch


def _surface_points_for_viewer(trajectory: ParallelismTrajectory) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = ParallelismCfg()
    root_pos = trajectory.root_pos_w[:, 0]
    root_rpy = trajectory.root_rpy_w[:, 0]
    joint = trajectory.joint_pos[:, 0]
    geometry = fk_go2(root_pos, root_rpy, joint)
    specs = tuple(cfg.official_collision_shapes)
    points_l, mask = build_official_surface_points_l(
        specs,
        cfg,
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    link_pos = []
    link_rot = []
    for spec in specs:
        if spec.link_type == "thigh":
            link_pos.append(geometry.thigh_pos_w)
            link_rot.append(geometry.thigh_rot_w)
        elif spec.link_type == "calf":
            link_pos.append(geometry.calf_pos_w)
            link_rot.append(geometry.calf_rot_w)
        elif spec.link_type == "foot":
            link_pos.append(geometry.foot_pos_w)
            link_rot.append(geometry.foot_rot_w)
        else:
            raise ValueError(f"unsupported collision link_type for viewer: {spec.link_type}")
    link_pos_t = torch.stack(link_pos, dim=2)
    link_rot_t = torch.stack(link_rot, dim=2)
    points_w = torch.matmul(link_rot_t[:, :, :, None], points_l.view(1, 1, len(specs), -1, 3, 1)).squeeze(-1)
    points_w = points_w + link_pos_t[:, :, :, None]
    points_w = points_w[mask.view(1, 1, len(specs), -1).expand_as(points_w[..., 0])]
    centers_w = link_pos_t.reshape(root_pos.shape[0], -1, 3)
    return points_w.reshape(-1, 3), centers_w


def parallelism_trajectory_to_viewer_result(trajectory: ParallelismTrajectory):
    root_quat_w = euler_to_quat_batch(
        trajectory.root_rpy_w[..., 0],
        trajectory.root_rpy_w[..., 1],
        trajectory.root_rpy_w[..., 2],
    )
    surface_points_w, collision_body_centers_w = _surface_points_for_viewer(trajectory)
    return SimpleNamespace(
        num_frames=int(trajectory.root_pos_w.shape[1]),
        root_pos_w=trajectory.root_pos_w,
        root_quat_w=root_quat_w,
        joint_angles=trajectory.joint_pos,
        foot_pos_w=trajectory.foot_pos_w,
        foot_pos_root=trajectory.foot_pos_w - trajectory.root_pos_w.unsqueeze(2),
        contact_state=trajectory.contact_state,
        planned_touchdown_w=trajectory.selected_foothold_w,
        parallelism_candidate_center_w=trajectory.diagnostics.candidate_center_w,
        parallelism_candidate_radius_m=float(trajectory.diagnostics.candidate_radius_m),
        parallelism_collision_surface_points_w=surface_points_w,
        parallelism_collision_body_centers_w=collision_body_centers_w,
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
