"""Continuous trajectory decode for the MPC-QP backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles, solve_joint_angles_from_trajectory
from extension.batch_mpc_planner.terrain import height_at
from extension.batch_mpc_planner.types import MpcPlannerResult, MpcPlannerTerrain

from .bezier import cubic_bezier_basis, sample_cubic_bezier


@dataclass(frozen=True)
class ContinuousTrajectoryControls:
    """Four-leg cubic controls with terrain-bound touchdown endpoints."""

    foot_control_w: Tensor
    root_pos_w: Tensor
    root_rpy: Tensor


def _phase_index(horizon: int, fraction: float) -> int:
    return max(0, min(horizon - 1, int(round(float(fraction) * float(max(horizon - 1, 1))))))


def build_controls_from_nominal(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    *,
    start_tangent_scale: float = 1.0,
) -> ContinuousTrajectoryControls:
    foot = torch.as_tensor(result.foot_pos)
    batch, horizon, legs = int(foot.shape[0]), int(foot.shape[1]), int(foot.shape[2])
    p0 = foot[:, 0].clone()
    p1 = foot[:, _phase_index(horizon, 1.0 / 3.0)]
    p1 = p0 + (p1 - p0) * float(start_tangent_scale)
    p1[..., 2] = torch.maximum(p1[..., 2], p0[..., 2] - 0.04)
    p2 = foot[:, _phase_index(horizon, 2.0 / 3.0)]
    p3 = torch.as_tensor(result.planned_touchdown_w[:, 0], dtype=foot.dtype, device=foot.device).clone()
    touchdown_z = height_at(terrain, p3[..., :2]).to(dtype=foot.dtype, device=foot.device)
    p3[..., 2] = touchdown_z
    controls = torch.stack((p0, p1, p2, p3), dim=2).reshape(batch, legs, 4, 3)
    return ContinuousTrajectoryControls(
        foot_control_w=controls,
        root_pos_w=_ease_root_start(result.root_pos, scale=float(start_tangent_scale)),
        root_rpy=result.root_rpy.clone(),
    )


def _ease_root_start(root_pos: Tensor, *, scale: float) -> Tensor:
    horizon = int(root_pos.shape[1])
    if horizon < 3:
        return root_pos.clone()
    eased = root_pos.clone()
    phase = torch.linspace(0.0, 1.0, horizon, dtype=root_pos.dtype, device=root_pos.device)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    linear = phase
    blend = float(scale) * linear + (1.0 - float(scale)) * smooth
    start = root_pos[:, :1, :]
    end = root_pos[:, -1:, :]
    eased = start + (end - start) * blend.view(1, horizon, 1)
    eased[..., 2] = root_pos[..., 2]
    return eased


def decode_controls_to_result(
    result: MpcPlannerResult,
    terrain: MpcPlannerTerrain,
    controls: ContinuousTrajectoryControls,
    *,
    sample_count: int,
    contact_state: Tensor | None = None,
) -> MpcPlannerResult:
    controls_w = torch.as_tensor(controls.foot_control_w, dtype=result.foot_pos.dtype, device=result.foot_pos.device).clone()
    touchdown_z = height_at(terrain, controls_w[:, :, 3, :2]).to(dtype=controls_w.dtype, device=controls_w.device)
    controls_w[:, :, 3, 2] = touchdown_z
    foot_pos = sample_controls_with_optional_gait(controls_w, sample_count=int(sample_count), contact_state=contact_state)
    horizon = int(foot_pos.shape[1])
    if horizon != int(result.root_pos.shape[1]):
        raise ValueError(f"sample_count must match result horizon {int(result.root_pos.shape[1])}, got {horizon}")
    root_pos = torch.as_tensor(controls.root_pos_w, dtype=result.root_pos.dtype, device=result.root_pos.device).clone()
    root_rpy = torch.as_tensor(controls.root_rpy, dtype=result.root_rpy.dtype, device=result.root_rpy.device).clone()
    joint_angles = solve_joint_angles_from_trajectory(root_pos, root_rpy, foot_pos)
    foot_pos = fk_feet_from_joint_angles(root_pos, root_rpy, joint_angles)
    planned_touchdown_w = result.planned_touchdown_w.clone()
    planned_touchdown_w[..., :2] = controls_w[:, None, :, 3, :2].expand_as(planned_touchdown_w[..., :2])
    planned_touchdown_w[..., 2] = controls_w[:, None, :, 3, 2].expand_as(planned_touchdown_w[..., 2])
    touchdown_seq = result.touchdown_seq.clone()
    event_cap = int(touchdown_seq.shape[2])
    touchdown_seq[..., :2] = controls_w[:, :, None, 3, :2].expand(-1, -1, event_cap, -1)
    touchdown_seq[..., 2] = controls_w[:, :, None, 3, 2].expand(-1, -1, event_cap)
    return MpcPlannerResult(
        root_pos=root_pos,
        root_rpy=root_rpy,
        foot_pos=foot_pos,
        joint_angles=joint_angles,
        contact_state=result.contact_state if contact_state is None else contact_state.to(dtype=torch.bool, device=result.contact_state.device),
        touchdown_seq=touchdown_seq,
        planned_touchdown_w=planned_touchdown_w,
        cost_total=result.cost_total,
        cost_breakdown=result.cost_breakdown,
        status=result.status,
        feasible=result.feasible,
        safe_fallback=result.safe_fallback,
        loss_breakdown=result.loss_breakdown,
        hard_reason_mask=result.hard_reason_mask,
    )


def sample_controls_with_optional_gait(
    controls_w: Tensor,
    *,
    sample_count: int,
    contact_state: Tensor | None = None,
) -> Tensor:
    basis = cubic_bezier_basis(int(sample_count), dtype=controls_w.dtype, device=controls_w.device)
    foot_pos = sample_cubic_bezier(controls_w, basis).transpose(1, 2).contiguous()
    if contact_state is None:
        return foot_pos
    horizon = int(foot_pos.shape[1])
    split = max(1, horizon // 2)
    gait_foot = foot_pos.clone()
    if split + 1 <= horizon:
        first_basis = cubic_bezier_basis(split + 1, dtype=controls_w.dtype, device=controls_w.device)
        first_samples = sample_cubic_bezier(controls_w[:, (1, 2)], first_basis).transpose(1, 2).contiguous()
        gait_foot[:, : split + 1, (1, 2), :] = first_samples
        touchdown = controls_w[:, (1, 2), 3, :].view(controls_w.shape[0], 1, 2, 3)
        gait_foot[:, split:, (1, 2), :] = touchdown.expand(-1, horizon - split, -1, -1)
    second_count = horizon - split
    if second_count > 0:
        second_basis = cubic_bezier_basis(second_count, dtype=controls_w.dtype, device=controls_w.device)
        second_samples = sample_cubic_bezier(controls_w[:, (0, 3)], second_basis).transpose(1, 2).contiguous()
        start_anchor = controls_w[:, (0, 3), 0, :].view(controls_w.shape[0], 1, 2, 3)
        gait_foot[:, :split, (0, 3), :] = start_anchor.expand(-1, split, -1, -1)
        gait_foot[:, split:, (0, 3), :] = second_samples
    return gait_foot


__all__ = [
    "ContinuousTrajectoryControls",
    "build_controls_from_nominal",
    "decode_controls_to_result",
    "sample_controls_with_optional_gait",
]
