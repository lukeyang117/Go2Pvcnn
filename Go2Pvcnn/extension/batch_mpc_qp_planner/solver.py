"""Fixed-shape continuous control updates for MPC-QP."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.batch_mpc_planner.kinematics import (
    CALF_LENGTH,
    HIP_OFFSETS_ARRAY,
    THIGH_LENGTH,
    fk_feet_from_joint_angles,
    fk_leg_points_from_joint_angles,
    _rpy_to_rot_matrix,
    solve_joint_angles_from_trajectory,
)
from extension.batch_mpc_planner.terrain import height_at, semantic_at
from extension.batch_mpc_planner.types import MpcPlannerTerrain

from .bezier import cubic_bezier_basis, sample_cubic_bezier
from .config import MpcQpPlannerCfg
from .continuous import ContinuousTrajectoryControls, sample_controls_with_optional_gait
from .coupled_solver import coupled_qp_update
from .fields import build_qp_fields
from .losses import _footprint_offsets

_JOINT_LIMITS = torch.tensor(
    (
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-1.5708, 3.4907),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
        (-1.0472, 1.0472),
        (-0.5236, 4.5379),
        (-2.7227, -0.8378),
    ),
    dtype=torch.float32,
)


def _phase_index(horizon: int, fraction: float) -> int:
    return max(0, min(horizon - 1, int(round(float(fraction) * float(max(horizon - 1, 1))))))


def _footprint_height_variation(terrain: MpcPlannerTerrain, touchdown_xy: Tensor, *, radius_m: float) -> Tensor:
    offsets = _footprint_offsets(radius_m=radius_m, dtype=touchdown_xy.dtype, device=touchdown_xy.device)
    points = touchdown_xy[..., None, :] + offsets.view(1, 1, -1, 2)
    height = height_at(terrain, points.reshape(touchdown_xy.shape[0], -1, 2)).reshape(*points.shape[:-1])
    return height.amax(dim=-1) - height.amin(dim=-1)


def _terrain_bound_controls(controls: Tensor, terrain: MpcPlannerTerrain) -> Tensor:
    out = controls.clone()
    touchdown_z = height_at(terrain, out[:, :, 3, :2]).to(dtype=out.dtype, device=out.device)
    out[:, :, 3, 2] = touchdown_z
    return out


def _underbody_points(root_pos: Tensor, *, sample_count: int = 5) -> Tensor:
    batch, horizon = int(root_pos.shape[0]), int(root_pos.shape[1])
    offsets = torch.tensor(
        (
            (0.0, 0.0, -0.16),
            (0.18, 0.10, -0.16),
            (0.18, -0.10, -0.16),
            (-0.18, 0.10, -0.16),
            (-0.18, -0.10, -0.16),
        ),
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    count = max(1, min(int(sample_count), int(offsets.shape[0])))
    return root_pos[:, :, None, :] + offsets[:count].view(1, 1, count, 3).expand(batch, horizon, -1, -1)


def _body_support_height_at(
    terrain: MpcPlannerTerrain,
    xy: Tensor,
    *,
    baseline_h: Tensor,
    low_small_height_m: float,
    probe_m: float = 0.04,
) -> Tensor:
    terrain_h = height_at(terrain, xy).to(dtype=xy.dtype, device=xy.device)
    while baseline_h.ndim < terrain_h.ndim:
        baseline_h = baseline_h.unsqueeze(-1)
    baseline_h = baseline_h.to(dtype=terrain_h.dtype, device=terrain_h.device)
    probe = float(probe_m)
    offsets = torch.tensor(
        ((0.0, 0.0), (probe, 0.0), (-probe, 0.0), (0.0, probe), (0.0, -probe)),
        dtype=xy.dtype,
        device=xy.device,
    )
    probe_xy = xy.unsqueeze(-2) + offsets.view(*((1,) * (xy.ndim - 1)), 5, 2)
    semantic = semantic_at(terrain, probe_xy.reshape(xy.shape[0], -1, 2)).reshape(*probe_xy.shape[:-1])
    nearby_low_small = (semantic == 1).any(dim=-1)
    low_small = torch.logical_and(nearby_low_small, (terrain_h - baseline_h) <= float(low_small_height_m))
    return torch.where(low_small, baseline_h.expand_as(terrain_h), terrain_h)


def _swing_clearance_lift(
    controls: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[Tensor, dict[str, Tensor]]:
    sample_count = max(5, int(cfg.runtime.horizon_steps))
    basis = cubic_bezier_basis(sample_count, dtype=controls.dtype, device=controls.device)
    samples = sample_cubic_bezier(controls, basis)
    semantic = semantic_at(terrain, samples[..., :2].reshape(samples.shape[0], -1, 2)).reshape(*samples.shape[:-1])
    low_small = semantic == 1
    terrain_z = height_at(terrain, samples[..., :2].reshape(samples.shape[0], -1, 2)).reshape(*samples.shape[:-1])
    terrain_z = terrain_z.to(dtype=samples.dtype, device=samples.device)
    required = terrain_z + float(cfg.runtime.low_small_swing_clearance_m)
    deficit = torch.clamp(required - samples[..., 2], min=0.0)
    phase = torch.linspace(0.0, 1.0, sample_count, dtype=controls.dtype, device=controls.device)
    mid_swing = torch.logical_and(phase > 0.05, phase < 0.95).view(1, 1, -1)
    active = torch.logical_and(low_small, mid_swing)
    masked_deficit = torch.where(active, deficit, torch.zeros_like(deficit))
    clearance_max = max(
        float(getattr(cfg.runtime, "low_small_swing_clearance_max_m", 0.16)),
        float(cfg.runtime.low_small_swing_clearance_m) + 0.04,
    )
    excess = torch.clamp(samples[..., 2] - (terrain_z + clearance_max), min=0.0)
    masked_excess = torch.where(active, excess, torch.zeros_like(excess))
    target_delta = masked_deficit - masked_excess
    active_f = active.to(dtype=controls.dtype, device=controls.device)
    b1 = basis[:, 1].view(1, 1, -1)
    b2 = basis[:, 2].view(1, 1, -1)
    denom1 = (b1.square() * active_f).sum(dim=-1).clamp_min(1.0e-6)
    denom2 = (b2.square() * active_f).sum(dim=-1).clamp_min(1.0e-6)
    p1_delta_z = (target_delta * b1).sum(dim=-1) / denom1
    p2_delta_z = (target_delta * b2).sum(dim=-1) / denom2
    lower_step = float(getattr(cfg.runtime, "low_small_swing_height_lower_step_m", 0.20))
    lift_step = float(cfg.runtime.low_small_swing_clearance_m)
    p1_delta_z = torch.clamp(p1_delta_z, min=-lower_step, max=lift_step)
    p2_delta_z = torch.clamp(p2_delta_z, min=-lower_step, max=lift_step)
    updated = controls.clone()
    updated[:, :, 1, 2] = updated[:, :, 1, 2] + p1_delta_z
    updated[:, :, 2, 2] = updated[:, :, 2, 2] + p2_delta_z
    diagnostics = {
        "qp_continuous_solver_swing_clearance_lift_count": torch.count_nonzero(
            torch.logical_or(p1_delta_z > 1.0e-6, p2_delta_z > 1.0e-6).reshape(controls.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_swing_height_lower_count": torch.count_nonzero(
            torch.logical_or(p1_delta_z < -1.0e-6, p2_delta_z < -1.0e-6).reshape(controls.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_swing_clearance_deficit_before_max": masked_deficit.reshape(
            masked_deficit.shape[0],
            -1,
        ).amax(dim=1),
        "qp_continuous_solver_swing_clearance_excess_before_max": masked_excess.reshape(
            masked_excess.shape[0],
            -1,
        ).amax(dim=1),
    }
    return updated, diagnostics


def _terrain_clearance_lift(
    controls: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    contact_state: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    sample_count = max(5, int(cfg.runtime.horizon_steps))
    basis = cubic_bezier_basis(sample_count, dtype=controls.dtype, device=controls.device)
    samples = sample_controls_with_optional_gait(
        controls,
        sample_count=sample_count,
        contact_state=contact_state,
    )
    terrain_z = height_at(terrain, samples[..., :2].reshape(samples.shape[0], -1, 2)).reshape(
        *samples.shape[:-1],
    )
    terrain_z = terrain_z.to(dtype=samples.dtype, device=samples.device)
    required = terrain_z + float(getattr(cfg.runtime, "continuous_terrain_clearance_m", 0.0))
    deficit = torch.clamp(required - samples[..., 2], min=0.0)
    if contact_state is None:
        active = torch.ones_like(deficit, dtype=torch.bool)
    else:
        active = torch.logical_not(contact_state.to(dtype=torch.bool, device=deficit.device))
    phase = torch.linspace(0.0, 1.0, sample_count, dtype=controls.dtype, device=controls.device)
    mid_swing = torch.logical_and(phase > 0.02, phase < 0.98).view(1, -1, 1)
    active = torch.logical_and(active, mid_swing)
    target_delta = torch.where(active, deficit, torch.zeros_like(deficit))
    active_f = active.to(dtype=controls.dtype, device=controls.device)
    coeff1 = basis[:, 1].view(1, sample_count, 1).expand(controls.shape[0], -1, controls.shape[1])
    coeff2 = basis[:, 2].view(1, sample_count, 1).expand(controls.shape[0], -1, controls.shape[1])
    if contact_state is not None:
        coeff1 = torch.zeros_like(target_delta)
        coeff2 = torch.zeros_like(target_delta)
        split = max(1, sample_count // 2)
        if split + 1 <= sample_count:
            first_basis = cubic_bezier_basis(split + 1, dtype=controls.dtype, device=controls.device)
            coeff1[:, : split + 1, (1, 2)] = first_basis[:, 1].view(1, split + 1, 1)
            coeff2[:, : split + 1, (1, 2)] = first_basis[:, 2].view(1, split + 1, 1)
        second_count = sample_count - split
        if second_count > 0:
            second_basis = cubic_bezier_basis(second_count, dtype=controls.dtype, device=controls.device)
            coeff1[:, split:, (0, 3)] = second_basis[:, 1].view(1, second_count, 1)
            coeff2[:, split:, (0, 3)] = second_basis[:, 2].view(1, second_count, 1)
    denom1 = (coeff1.square() * active_f).sum(dim=1).clamp_min(1.0e-6)
    denom2 = (coeff2.square() * active_f).sum(dim=1).clamp_min(1.0e-6)
    p1_delta_z = (target_delta * coeff1).sum(dim=1) / denom1
    p2_delta_z = (target_delta * coeff2).sum(dim=1) / denom2
    max_step = float(getattr(cfg.runtime, "continuous_terrain_clearance_step_m", 0.08))
    p1_delta_z = torch.clamp(p1_delta_z, min=0.0, max=max_step)
    p2_delta_z = torch.clamp(p2_delta_z, min=0.0, max=max_step)
    updated = controls.clone()
    updated[:, :, 1, 2] = updated[:, :, 1, 2] + p1_delta_z
    updated[:, :, 2, 2] = updated[:, :, 2, 2] + p2_delta_z
    diagnostics = {
        "qp_continuous_solver_terrain_clearance_lift_count": torch.count_nonzero(
            torch.logical_or(p1_delta_z > 1.0e-6, p2_delta_z > 1.0e-6).reshape(controls.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_terrain_clearance_deficit_before_max": target_delta.reshape(
            target_delta.shape[0],
            -1,
        ).amax(dim=1),
    }
    return updated, diagnostics


def _fk_readback_update(
    controls: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    contact_state: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    def _root_z_pass(current_controls: Tensor, current_root: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        target = sample_controls_with_optional_gait(
            current_controls,
            sample_count=sample_count,
            contact_state=contact_state,
        )
        joint = solve_joint_angles_from_trajectory(current_root, root_rpy, target)
        fk_foot_current = fk_feet_from_joint_angles(current_root, root_rpy, joint)
        z_residual = fk_foot_current[..., 2] - target[..., 2]
        positive_residual = torch.clamp(z_residual, min=0.0)
        frame_delta = positive_residual.amax(dim=-1) * root_z_gain
        frame_error = positive_residual.amax(dim=-1)
        frame_delta = torch.where(
            frame_error > root_z_error_threshold,
            frame_delta,
            torch.zeros_like(frame_delta),
        )
        target_semantic = semantic_at(terrain, target[..., :2])
        underbody = _underbody_points(current_root)
        underbody_semantic = semantic_at(terrain, underbody[..., :2])
        underbody_terrain = height_at(terrain, underbody[..., :2]).to(
            dtype=current_root.dtype,
            device=current_root.device,
        )
        underbody_clearance = underbody[..., 2] - (underbody_terrain + 0.015)
        no_semantic_limit = torch.full_like(underbody_clearance, root_z_max_step)
        underbody_lower_cap = torch.where(underbody_semantic != 0, underbody_clearance, no_semantic_limit)
        underbody_lower_cap = underbody_lower_cap.amin(dim=-1).clamp(min=0.0, max=root_z_max_step)
        target_terrain = height_at(terrain, target[..., :2]).to(dtype=current_root.dtype, device=current_root.device)
        terrain_variation = target_terrain.reshape(target_terrain.shape[0], -1).amax(dim=1) - target_terrain.reshape(
            target_terrain.shape[0],
            -1,
        ).amin(dim=1)
        row_flat_enough = terrain_variation <= root_z_terrain_variation_threshold
        row_allowed = row_flat_enough
        frame_delta = torch.where(row_allowed[:, None], frame_delta, torch.zeros_like(frame_delta))
        frame_delta = torch.minimum(frame_delta, underbody_lower_cap)
        frame_delta = frame_delta.clamp(max=root_z_max_step)
        ground = height_at(terrain, current_root[..., :2].reshape(current_root.shape[0], -1, 2)).reshape(
            current_root.shape[0],
            current_root.shape[1],
        )
        ground = ground.to(dtype=current_root.dtype, device=current_root.device)
        min_height = ground + root_z_min_offset
        next_root = current_root.clone()
        proposed_z = torch.maximum(current_root[..., 2] - frame_delta, min_height)
        next_root[..., 2] = torch.where(frame_delta > 1.0e-6, proposed_z, current_root[..., 2])
        actual_delta = torch.clamp(current_root[..., 2] - next_root[..., 2], min=0.0)
        return next_root, actual_delta, torch.linalg.vector_norm(fk_foot_current - target, dim=-1)

    def _root_xy_pass(current_controls: Tensor, current_root: Tensor) -> tuple[Tensor, Tensor]:
        if contact_state is None or root_xy_gain <= 0.0 or root_xy_max_step <= 0.0:
            return current_root, torch.zeros(current_root.shape[:2], dtype=current_root.dtype, device=current_root.device)
        target = sample_controls_with_optional_gait(
            current_controls,
            sample_count=sample_count,
            contact_state=contact_state,
        )
        joint = solve_joint_angles_from_trajectory(current_root, root_rpy, target)
        fk_foot_current = fk_feet_from_joint_angles(current_root, root_rpy, joint)
        contact = torch.as_tensor(contact_state, dtype=torch.bool, device=current_root.device)
        residual_xy = fk_foot_current[..., :2] - target[..., :2]
        residual_xy = torch.where(contact.unsqueeze(-1), residual_xy, torch.zeros_like(residual_xy))
        target_semantic = semantic_at(terrain, target[..., :2])
        target_terrain = height_at(terrain, target[..., :2]).to(dtype=current_root.dtype, device=current_root.device)
        terrain_variation = target_terrain.reshape(target_terrain.shape[0], -1).amax(dim=1) - target_terrain.reshape(
            target_terrain.shape[0],
            -1,
        ).amin(dim=1)
        semantic_present = (target_semantic != 0).reshape(target_semantic.shape[0], -1).any(dim=1)
        row_allowed = torch.logical_or(semantic_present, terrain_variation <= root_z_terrain_variation_threshold)
        residual_xy = torch.where(row_allowed[:, None, None, None], residual_xy, torch.zeros_like(residual_xy))
        contact_count = contact.to(dtype=current_root.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        frame_delta = residual_xy.sum(dim=2) / contact_count * root_xy_gain
        delta_norm = torch.linalg.vector_norm(frame_delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
        frame_delta = frame_delta / delta_norm * torch.clamp(delta_norm, max=root_xy_max_step)
        active = torch.linalg.vector_norm(frame_delta, dim=-1) > root_xy_error_threshold
        next_root = current_root.clone()
        next_root[..., :2] = next_root[..., :2] - torch.where(active.unsqueeze(-1), frame_delta, torch.zeros_like(frame_delta))
        return next_root, torch.linalg.vector_norm(frame_delta, dim=-1)

    def _one_pass(current_controls: Tensor, current_root: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        target = sample_controls_with_optional_gait(
            current_controls,
            sample_count=sample_count,
            contact_state=contact_state,
        )
        joint = solve_joint_angles_from_trajectory(current_root, root_rpy, target)
        fk_foot_current = fk_feet_from_joint_angles(current_root, root_rpy, joint)
        residual_current = fk_foot_current - target
        residual_norm_current = torch.linalg.vector_norm(residual_current, dim=-1)
        p1_delta_current = residual_current[:, p1_idx] * gain
        p2_delta_current = residual_current[:, p2_idx] * gain
        p3_delta_current = residual_current[:, -1, :, :2] * endpoint_gain
        p1_norm_current = torch.linalg.vector_norm(p1_delta_current, dim=-1, keepdim=True).clamp_min(1.0e-6)
        p2_norm_current = torch.linalg.vector_norm(p2_delta_current, dim=-1, keepdim=True).clamp_min(1.0e-6)
        p3_norm_current = torch.linalg.vector_norm(p3_delta_current, dim=-1, keepdim=True).clamp_min(1.0e-6)
        p1_delta_current = p1_delta_current / p1_norm_current * torch.clamp(p1_norm_current, max=max_step)
        p2_delta_current = p2_delta_current / p2_norm_current * torch.clamp(p2_norm_current, max=max_step)
        p3_delta_current = p3_delta_current / p3_norm_current * torch.clamp(p3_norm_current, max=endpoint_max_step)
        next_controls = current_controls.clone()
        next_controls[:, :, 1, :] = next_controls[:, :, 1, :] + p1_delta_current
        next_controls[:, :, 2, :] = next_controls[:, :, 2, :] + p2_delta_current
        next_controls[:, :, 1, :2] = next_controls[:, :, 1, :2] + p3_delta_current * (1.0 / 3.0)
        next_controls[:, :, 2, :2] = next_controls[:, :, 2, :2] + p3_delta_current * (2.0 / 3.0)
        next_controls[:, :, 3, :2] = next_controls[:, :, 3, :2] + p3_delta_current
        return (
            _terrain_bound_controls(next_controls, terrain),
            residual_norm_current,
            p1_delta_current,
            p2_delta_current,
            p3_delta_current,
        )

    sample_count = int(root_pos.shape[1])
    max_step = float(cfg.runtime.continuous_fk_readback_max_step_m)
    endpoint_max_step = float(getattr(cfg.runtime, "continuous_fk_endpoint_max_step_m", max_step))
    gain = float(cfg.runtime.continuous_fk_readback_gain)
    endpoint_gain = float(getattr(cfg.runtime, "continuous_fk_endpoint_gain", gain))
    root_z_gain = float(getattr(cfg.runtime, "continuous_fk_root_z_gain", 0.0))
    root_z_max_step = float(getattr(cfg.runtime, "continuous_fk_root_z_max_step_m", 0.0))
    root_z_error_threshold = float(getattr(cfg.runtime, "continuous_fk_root_z_error_threshold_m", 0.08))
    root_z_min_offset = float(getattr(cfg.runtime, "continuous_plane_root_height_min_m", 0.30))
    root_z_terrain_variation_threshold = float(getattr(cfg.runtime, "terrain_height_variation_threshold_m", 0.06))
    root_xy_gain = float(getattr(cfg.runtime, "continuous_fk_root_xy_gain", 0.0))
    root_xy_max_step = float(getattr(cfg.runtime, "continuous_fk_root_xy_max_step_m", 0.0))
    root_xy_error_threshold = 1.0e-3
    p1_idx = _phase_index(sample_count, 1.0 / 3.0)
    p2_idx = _phase_index(sample_count, 2.0 / 3.0)
    root_updated = root_pos
    if root_z_gain > 0.0 and root_z_max_step > 0.0:
        root_updated, root_z_delta, root_readback_norm = _root_z_pass(controls, root_updated)
    else:
        root_z_delta = torch.zeros(root_pos.shape[:2], dtype=root_pos.dtype, device=root_pos.device)
        root_readback_norm = torch.zeros((*root_pos.shape[:2], controls.shape[1]), dtype=root_pos.dtype, device=root_pos.device)
    root_updated, root_xy_delta = _root_xy_pass(controls, root_updated)
    updated, residual_norm, p1_delta, p2_delta, p3_delta_xy = _one_pass(controls, root_updated)
    updated, _, p1_delta_next, p2_delta_next, p3_delta_xy_next = _one_pass(updated, root_updated)
    p1_delta = p1_delta + p1_delta_next
    p2_delta = p2_delta + p2_delta_next
    p3_delta_xy = p3_delta_xy + p3_delta_xy_next
    endpoint_updated = torch.linalg.vector_norm(p3_delta_xy, dim=-1) > 1.0e-6
    if contact_state is not None:
        target = sample_controls_with_optional_gait(
            updated,
            sample_count=sample_count,
            contact_state=contact_state,
        )
        joint = solve_joint_angles_from_trajectory(root_updated, root_rpy, target)
        fk_foot = fk_feet_from_joint_angles(root_updated, root_rpy, joint)
        residual = fk_foot - target
        split = max(1, sample_count // 2)
        segment_delta = torch.zeros_like(updated[:, :, 1, :])
        if split + 1 <= sample_count:
            segment_delta[:, (1, 2), :] = residual[:, : split + 1, (1, 2), :].mean(dim=1)
        if sample_count - split > 0:
            segment_delta[:, (0, 3), :] = residual[:, split:, (0, 3), :].mean(dim=1)
        segment_norm = torch.linalg.vector_norm(segment_delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
        segment_delta = segment_delta / segment_norm * torch.clamp(segment_norm, max=max_step)
        segment_delta[..., 2] = 0.0
        segment_active = torch.linalg.vector_norm(segment_delta, dim=-1) > 1.0e-6
        updated = updated.clone()
        updated[:, :, 1, :] = updated[:, :, 1, :] + segment_delta * (1.0 / 3.0)
        updated[:, :, 2, :] = updated[:, :, 2, :] + segment_delta * (2.0 / 3.0)
        updated[:, :, 3, :2] = updated[:, :, 3, :2] + segment_delta[..., :2]
        updated = _terrain_bound_controls(updated, terrain)
    else:
        segment_active = torch.zeros_like(endpoint_updated)
    diagnostics = {
        "qp_continuous_solver_fk_readback_update_count": torch.count_nonzero(
            torch.logical_or(
                torch.logical_or(
                    torch.linalg.vector_norm(p1_delta, dim=-1) > 1.0e-6,
                    torch.linalg.vector_norm(p2_delta, dim=-1) > 1.0e-6,
                ),
                torch.logical_or(endpoint_updated, segment_active),
            ).reshape(controls.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_fk_segment_readback_update_count": torch.count_nonzero(
            segment_active.reshape(controls.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_fk_endpoint_update_count": torch.count_nonzero(
            endpoint_updated.reshape(controls.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_fk_readback_error_before_max": residual_norm.reshape(controls.shape[0], -1).amax(dim=1),
        "qp_continuous_solver_fk_endpoint_error_before_max": residual_norm[:, -1].reshape(controls.shape[0], -1).amax(dim=1),
        "qp_continuous_solver_fk_root_z_update_count": torch.count_nonzero(
            root_z_delta > 1.0e-6,
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_fk_root_z_delta_max": root_z_delta.reshape(controls.shape[0], -1).amax(dim=1).to(
            dtype=controls.dtype,
        ),
        "qp_continuous_solver_fk_root_xy_update_count": torch.count_nonzero(
            root_xy_delta > root_xy_error_threshold,
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_fk_root_xy_delta_max": root_xy_delta.reshape(controls.shape[0], -1).amax(dim=1).to(
            dtype=controls.dtype,
        ),
        "qp_continuous_solver_fk_root_readback_error_before_max": root_readback_norm.reshape(
            controls.shape[0],
            -1,
        ).amax(dim=1).to(dtype=controls.dtype),
    }
    return updated, root_updated, diagnostics


def _joint_limit_readback_update(
    controls: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    contact_state: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    sample_count = int(root_pos.shape[1])
    target = sample_controls_with_optional_gait(
        controls,
        sample_count=sample_count,
        contact_state=contact_state,
    )
    raw_joint = solve_joint_angles_from_trajectory(root_pos, root_rpy, target, clamp_to_limits=False)
    clamped_joint = solve_joint_angles_from_trajectory(root_pos, root_rpy, target, clamp_to_limits=True)
    limits = _JOINT_LIMITS.to(device=root_pos.device, dtype=root_pos.dtype).view(1, 1, 4, 3, 2)
    raw_leg = raw_joint.reshape(root_pos.shape[0], sample_count, 4, 3)
    violation = torch.maximum(
        torch.relu(limits[..., 0] - raw_leg),
        torch.relu(raw_leg - limits[..., 1]),
    ).amax(dim=-1)
    semantic = semantic_at(terrain, target[..., :2].reshape(target.shape[0], -1, 2)).reshape(*target.shape[:-1])
    active = torch.logical_and(violation > 1.0e-5, semantic == 1)
    if not bool(torch.any(active).item()):
        zero = torch.zeros((controls.shape[0],), dtype=controls.dtype, device=controls.device)
        return controls, {
            "qp_continuous_solver_joint_limit_readback_update_count": zero,
            "qp_continuous_solver_joint_limit_violation_before_max": violation.reshape(controls.shape[0], -1).amax(dim=1).to(dtype=controls.dtype),
        }
    fk_foot = fk_feet_from_joint_angles(root_pos, root_rpy, clamped_joint)
    residual = fk_foot - target
    p1_idx = _phase_index(sample_count, 1.0 / 3.0)
    p2_idx = _phase_index(sample_count, 2.0 / 3.0)
    gain = float(getattr(cfg.runtime, "continuous_joint_limit_readback_gain", 0.75))
    max_step = float(getattr(cfg.runtime, "continuous_joint_limit_readback_max_step_m", 0.10))
    endpoint_gain = float(getattr(cfg.runtime, "continuous_joint_limit_endpoint_gain", 0.35))
    endpoint_max_step = float(getattr(cfg.runtime, "continuous_joint_limit_endpoint_max_step_m", 0.10))
    p1_delta = residual[:, p1_idx] * gain
    p2_delta = residual[:, p2_idx] * gain
    p3_delta = residual[:, -1, :, :2] * endpoint_gain
    p1_delta = torch.where(active[:, p1_idx].unsqueeze(-1), p1_delta, torch.zeros_like(p1_delta))
    p2_delta = torch.where(active[:, p2_idx].unsqueeze(-1), p2_delta, torch.zeros_like(p2_delta))
    p3_delta = torch.where(active[:, -1].unsqueeze(-1), p3_delta, torch.zeros_like(p3_delta))

    def _cap(delta: Tensor, cap: float) -> Tensor:
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
        return delta / norm * torch.clamp(norm, max=float(cap))

    p1_delta = _cap(p1_delta, max_step)
    p2_delta = _cap(p2_delta, max_step)
    p3_delta = _cap(p3_delta, endpoint_max_step)
    updated = controls.clone()
    updated[:, :, 1, :] = updated[:, :, 1, :] + p1_delta
    updated[:, :, 2, :] = updated[:, :, 2, :] + p2_delta
    updated[:, :, 1, :2] = updated[:, :, 1, :2] + p3_delta * (1.0 / 3.0)
    updated[:, :, 2, :2] = updated[:, :, 2, :2] + p3_delta * (2.0 / 3.0)
    updated[:, :, 3, :2] = updated[:, :, 3, :2] + p3_delta
    updated = _terrain_bound_controls(updated, terrain)
    diagnostics = {
        "qp_continuous_solver_joint_limit_readback_update_count": torch.count_nonzero(
            torch.logical_or(
                torch.logical_or(torch.linalg.vector_norm(p1_delta, dim=-1) > 1.0e-6, torch.linalg.vector_norm(p2_delta, dim=-1) > 1.0e-6),
                torch.linalg.vector_norm(p3_delta, dim=-1) > 1.0e-6,
            ).reshape(controls.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_joint_limit_violation_before_max": violation.reshape(controls.shape[0], -1).amax(dim=1).to(dtype=controls.dtype),
    }
    return updated, diagnostics


def _touchdown_field_qp_step(
    touchdown_xy: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[Tensor, dict[str, Tensor]]:
    fields = build_qp_fields(terrain, eps_m=float(cfg.runtime.continuous_foothold_probe_m))
    sample = fields.query(touchdown_xy)
    target = float(cfg.runtime.continuous_foothold_variation_target_m)
    semantic_violation = torch.relu(sample.semantic_risk)
    roughness_violation = torch.relu(sample.roughness - target)
    grad = (
        sample.semantic_grad_xy * semantic_violation.unsqueeze(-1) * 1000.0
        + sample.roughness_grad_xy * roughness_violation.unsqueeze(-1)
        + sample.height_grad_xy * roughness_violation.unsqueeze(-1)
    )
    h_diag = 1.0 + 1000.0 * sample.semantic_grad_xy.square().sum(dim=-1, keepdim=True)
    h_diag = h_diag + sample.roughness_grad_xy.square().sum(dim=-1, keepdim=True)
    h_diag = h_diag + sample.height_grad_xy.square().sum(dim=-1, keepdim=True)
    raw_delta = -grad / h_diag.clamp_min(1.0e-6)
    max_step = float(cfg.runtime.continuous_foothold_step_m)
    norm = torch.linalg.vector_norm(raw_delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
    delta = raw_delta / norm * torch.clamp(norm, max=max_step)
    active = torch.logical_or(semantic_violation > 1.0e-6, roughness_violation > 1.0e-6)
    delta = torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))
    updated_xy = touchdown_xy + delta
    after = fields.query(updated_xy)
    diagnostics = {
        "qp_continuous_solver_update_count": torch.count_nonzero(active.reshape(active.shape[0], -1), dim=1).to(
            dtype=touchdown_xy.dtype,
        ),
        "qp_continuous_solver_foothold_score_before_max": sample.roughness.reshape(sample.roughness.shape[0], -1).amax(dim=1).to(dtype=touchdown_xy.dtype),
        "qp_continuous_solver_foothold_score_after_max": after.roughness.reshape(after.roughness.shape[0], -1).amax(dim=1).to(dtype=touchdown_xy.dtype),
        "qp_continuous_solver_semantic_score_before_max": sample.semantic_risk.reshape(sample.semantic_risk.shape[0], -1).amax(dim=1).to(dtype=touchdown_xy.dtype),
        "qp_continuous_solver_semantic_score_after_max": after.semantic_risk.reshape(after.semantic_risk.shape[0], -1).amax(dim=1).to(dtype=touchdown_xy.dtype),
    }
    return updated_xy, diagnostics


def _root_terrain_progress_update(
    root_pos: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[Tensor, dict[str, Tensor]]:
    sample_count = max(2, int(cfg.runtime.terrain_step_cap_sample_count))
    idx = torch.linspace(0, root_pos.shape[1] - 1, sample_count, device=root_pos.device).round().to(dtype=torch.long)
    sampled_xy = root_pos.index_select(1, idx)[..., :2]
    sampled_h = height_at(terrain, sampled_xy.reshape(root_pos.shape[0], -1, 2)).reshape(root_pos.shape[0], sample_count)
    sampled_semantic = semantic_at(terrain, sampled_xy.reshape(root_pos.shape[0], -1, 2)).reshape(
        root_pos.shape[0],
        sample_count,
    )
    baseline_h = sampled_h.amin(dim=1, keepdim=True)
    low_small_height = float(getattr(cfg.losses.low_small_crossing, "high_small_relative_height_m", 0.30))
    low_small = torch.logical_and(sampled_semantic == 1, (sampled_h - baseline_h) <= low_small_height)
    variation_h = torch.where(low_small, baseline_h.expand_as(sampled_h), sampled_h)
    variation = variation_h.amax(dim=1) - variation_h.amin(dim=1)
    threshold = float(cfg.runtime.terrain_height_variation_threshold_m)
    risky = variation > threshold
    min_scale = float(cfg.runtime.terrain_step_cap_min_scale)
    over = torch.clamp((variation - threshold) / max(threshold, 1.0e-6), min=0.0, max=1.0)
    scale = torch.where(
        risky,
        1.0 - (1.0 - min_scale) * over.to(dtype=root_pos.dtype, device=root_pos.device),
        torch.ones_like(variation, dtype=root_pos.dtype, device=root_pos.device),
    )
    root0 = root_pos[:, :1, :]
    updated = root_pos.clone()
    updated[..., :2] = root0[..., :2] + (root_pos[..., :2] - root0[..., :2]) * scale[:, None, None]
    root0_ground = height_at(terrain, root0[..., :2]).reshape(root_pos.shape[0], 1).to(
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    height_offset = (root0[..., 2] - root0_ground).clamp(0.26, 0.42)
    updated_ground = _body_support_height_at(
        terrain,
        updated[..., :2],
        baseline_h=baseline_h,
        low_small_height_m=low_small_height,
        probe_m=float(getattr(cfg.runtime, "continuous_foothold_probe_m", 0.04)),
    ).to(dtype=root_pos.dtype, device=root_pos.device)
    updated[..., 2] = updated_ground + height_offset
    plane_mask = getattr(terrain, "is_plane_terrain", None)
    plane_active = torch.zeros((root_pos.shape[0],), dtype=torch.bool, device=root_pos.device)
    if plane_mask is not None:
        plane_active = torch.as_tensor(plane_mask, dtype=torch.bool, device=root_pos.device).reshape(-1)[: root_pos.shape[0]]
    min_h = float(getattr(cfg.runtime, "continuous_plane_root_height_min_m", 0.30))
    max_h = float(getattr(cfg.runtime, "continuous_plane_root_height_max_m", 0.36))
    plane_height_offset = height_offset.clamp(min_h, max_h)
    updated[..., 2] = torch.where(
        plane_active[:, None],
        updated_ground + plane_height_offset,
        updated[..., 2],
    )
    diagnostics = {
        "qp_continuous_root_terrain_risk_reduces_progress": risky.to(dtype=root_pos.dtype, device=root_pos.device),
        "qp_continuous_root_height_variation_max": variation.to(dtype=root_pos.dtype, device=root_pos.device),
        "qp_continuous_root_progress_scale_min": scale.to(dtype=root_pos.dtype, device=root_pos.device),
        "qp_continuous_plane_root_height_clamp_count": torch.logical_and(
            plane_active,
            torch.abs(plane_height_offset.squeeze(-1) - height_offset.squeeze(-1)) > 1.0e-6,
        ).to(dtype=root_pos.dtype, device=root_pos.device),
    }
    return updated, diagnostics


def _root_attitude_level_update(
    root_rpy: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[Tensor, dict[str, Tensor]]:
    batch = int(root_rpy.shape[0])
    zero = torch.zeros((batch,), dtype=root_rpy.dtype, device=root_rpy.device)
    terrain_h = torch.as_tensor(terrain.height_map, dtype=root_rpy.dtype, device=root_rpy.device)
    semantic = torch.as_tensor(terrain.semantic_map, device=root_rpy.device)
    terrain_range = terrain_h.reshape(batch, -1).amax(dim=1) - terrain_h.reshape(batch, -1).amin(dim=1)
    has_low_small = semantic.reshape(batch, -1).eq(1).any(dim=1)
    plane = torch.as_tensor(terrain.is_plane_terrain, dtype=torch.bool, device=root_rpy.device).reshape(batch)
    active = torch.logical_and(
        torch.logical_and(plane, has_low_small),
        terrain_range <= float(getattr(cfg.runtime, "terrain_height_variation_threshold_m", 0.06)) + 0.20,
    )
    roll_pitch_abs = root_rpy[..., :2].abs().amax(dim=(1, 2))
    active = torch.logical_and(active, roll_pitch_abs > 1.0e-4)
    updated = root_rpy
    diagnostics = {
        "qp_continuous_solver_root_attitude_level_count": zero,
        "qp_continuous_solver_root_attitude_before_max": roll_pitch_abs.to(dtype=root_rpy.dtype),
    }
    return updated, diagnostics


def _low_small_crossing_progress_update(
    root_pos: Tensor,
    controls: Tensor,
    terrain: MpcPlannerTerrain,
    command: Tensor | None,
    cfg: MpcQpPlannerCfg,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    batch = int(root_pos.shape[0])
    if command is None:
        zero = torch.zeros((batch,), dtype=root_pos.dtype, device=root_pos.device)
        return root_pos, controls, {
            "qp_continuous_low_small_progress_update_count": zero,
            "qp_continuous_low_small_progress_deficit_before_max": zero,
        }
    cmd = torch.as_tensor(command, dtype=root_pos.dtype, device=root_pos.device)
    if cmd.ndim == 1:
        cmd = cmd.view(1, -1).expand(batch, -1)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    heading = cmd[:, :2]
    speed = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    active_cmd = speed.squeeze(-1) > 1.0e-5
    heading = heading / speed.clamp_min(1.0e-6)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    sample_count = max(3, int(getattr(cfg.runtime, "continuous_low_small_progress_sample_count", 17)))
    lookahead = float(getattr(cfg.runtime, "continuous_low_small_progress_lookahead_m", 0.48))
    lane = float(getattr(cfg.runtime, "continuous_low_small_progress_lane_half_width_m", 0.14))
    margin = float(getattr(cfg.runtime, "continuous_low_small_progress_margin_m", 0.06))
    along_samples = torch.linspace(0.0, lookahead, sample_count, dtype=root_pos.dtype, device=root_pos.device)
    lateral_samples = torch.tensor((-lane, 0.0, lane), dtype=root_pos.dtype, device=root_pos.device)
    root0_xy = root_pos[:, 0, :2]
    points = (
        root0_xy[:, None, None, :]
        + along_samples.view(1, sample_count, 1, 1) * heading[:, None, None, :]
        + lateral_samples.view(1, 1, 3, 1) * left[:, None, None, :]
    )
    semantic = semantic_at(terrain, points.reshape(batch, sample_count * 3, 2)).reshape(batch, sample_count, 3)
    height = height_at(terrain, points.reshape(batch, sample_count * 3, 2)).reshape(batch, sample_count, 3).to(
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    root0_ground = height_at(terrain, root0_xy[:, None, :]).reshape(batch, 1, 1).to(
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    low_small_height = float(getattr(cfg.losses.low_small_crossing, "high_small_relative_height_m", 0.30))
    low_small = torch.logical_and(semantic == 1, (height - root0_ground) <= low_small_height)
    any_low_small = torch.logical_and(low_small.any(dim=(1, 2)), active_cmd)
    masked_along = torch.where(
        low_small,
        along_samples.view(1, sample_count, 1).expand(batch, -1, 3),
        torch.full((batch, sample_count, 3), lookahead + 1.0, dtype=root_pos.dtype, device=root_pos.device),
    )
    obstacle_along = masked_along.reshape(batch, -1).amin(dim=1)
    desired_end_along = obstacle_along + margin
    current_end_along = ((root_pos[:, -1, :2] - root0_xy) * heading).sum(dim=-1)
    deficit = torch.where(any_low_small, torch.relu(desired_end_along - current_end_along), torch.zeros_like(current_end_along))
    max_step = float(getattr(cfg.runtime, "continuous_low_small_progress_step_m", 0.16))
    progress_delta = torch.clamp(deficit, min=0.0, max=max_step)
    delta_xy = progress_delta[:, None] * heading
    phase = torch.linspace(0.0, 1.0, int(root_pos.shape[1]), dtype=root_pos.dtype, device=root_pos.device)
    root_updated = root_pos.clone()
    root_updated[..., :2] = root_updated[..., :2] + phase.view(1, -1, 1) * delta_xy[:, None, :]
    ground = _body_support_height_at(
        terrain,
        root_updated[..., :2],
        baseline_h=root0_ground.reshape(batch),
        low_small_height_m=low_small_height,
        probe_m=float(getattr(cfg.runtime, "continuous_foothold_probe_m", 0.04)),
    ).to(dtype=root_pos.dtype, device=root_pos.device)
    height_offset = (root_pos[..., 2] - height_at(terrain, root_pos[..., :2]).to(dtype=root_pos.dtype, device=root_pos.device)).clamp(
        0.26,
        0.42,
    )
    root_updated[..., 2] = ground + height_offset
    controls_updated = controls.clone()
    controls_updated[:, :, 1, :2] = controls_updated[:, :, 1, :2] + delta_xy[:, None, :] * (1.0 / 3.0)
    controls_updated[:, :, 2, :2] = controls_updated[:, :, 2, :2] + delta_xy[:, None, :] * (2.0 / 3.0)
    controls_updated[:, :, 3, :2] = controls_updated[:, :, 3, :2] + delta_xy[:, None, :]
    controls_updated = _terrain_bound_controls(controls_updated, terrain)
    diagnostics = {
        "qp_continuous_low_small_progress_update_count": (progress_delta > 1.0e-6).to(dtype=root_pos.dtype),
        "qp_continuous_low_small_progress_deficit_before_max": deficit.to(dtype=root_pos.dtype),
        "qp_continuous_low_small_progress_obstacle_along_min": torch.where(
            any_low_small,
            obstacle_along,
            torch.zeros_like(obstacle_along),
        ).to(dtype=root_pos.dtype),
    }
    return root_updated, controls_updated, diagnostics


def _low_small_foot_over_update(
    controls: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    terrain: MpcPlannerTerrain,
    command: Tensor | None,
    cfg: MpcQpPlannerCfg,
    contact_state: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    del root_pos, root_rpy
    batch = int(controls.shape[0])
    zero = torch.zeros((batch,), dtype=controls.dtype, device=controls.device)
    if command is None:
        return controls, {
            "qp_continuous_low_small_foot_over_update_count": zero,
            "qp_continuous_low_small_foot_over_lateral_deficit_max": zero,
            "qp_continuous_low_small_foot_over_reach_reject_count": zero,
        }
    cmd = torch.as_tensor(command, dtype=controls.dtype, device=controls.device)
    if cmd.ndim == 1:
        cmd = cmd.view(1, -1).expand(batch, -1)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    heading = cmd[:, :2]
    speed = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    active_cmd = speed.squeeze(-1) > 1.0e-5
    heading = heading / speed.clamp_min(1.0e-6)
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    sample_count = max(3, int(getattr(cfg.runtime, "continuous_low_small_progress_sample_count", 17)))
    lookahead = float(getattr(cfg.runtime, "continuous_low_small_progress_lookahead_m", 0.48))
    lane = float(getattr(cfg.runtime, "continuous_low_small_progress_lane_half_width_m", 0.14))
    root_xy = controls[:, :, 0, :2].mean(dim=1)
    along_samples = torch.linspace(0.0, lookahead, sample_count, dtype=controls.dtype, device=controls.device)
    lateral_samples = torch.tensor((-lane, 0.0, lane), dtype=controls.dtype, device=controls.device)
    points = (
        root_xy[:, None, None, :]
        + along_samples.view(1, sample_count, 1, 1) * heading[:, None, None, :]
        + lateral_samples.view(1, 1, 3, 1) * left[:, None, None, :]
    )
    semantic = semantic_at(terrain, points.reshape(batch, sample_count * 3, 2)).reshape(batch, sample_count, 3)
    height = height_at(terrain, points.reshape(batch, sample_count * 3, 2)).reshape(batch, sample_count, 3).to(
        dtype=controls.dtype,
        device=controls.device,
    )
    root_ground = height_at(terrain, root_xy[:, None, :]).reshape(batch, 1, 1).to(
        dtype=controls.dtype,
        device=controls.device,
    )
    low_small_height = float(getattr(cfg.losses.low_small_crossing, "high_small_relative_height_m", 0.30))
    low_small = torch.logical_and(semantic == 1, (height - root_ground) <= low_small_height)
    active = torch.logical_and(low_small.any(dim=(1, 2)), active_cmd)
    obstacle_height = torch.where(low_small, height, root_ground.expand_as(height)).reshape(batch, -1).amax(dim=1)
    masked_along = torch.where(
        low_small,
        along_samples.view(1, sample_count, 1).expand(batch, -1, 3),
        torch.full((batch, sample_count, 3), lookahead + 1.0, dtype=controls.dtype, device=controls.device),
    )
    obstacle_along = masked_along.reshape(batch, -1).amin(dim=1)
    obstacle_xy = root_xy + obstacle_along[:, None] * heading
    p0_rel = controls[:, :, 0, :2] - obstacle_xy[:, None, :]
    p3_rel = controls[:, :, 3, :2] - obstacle_xy[:, None, :]
    p0_along = (p0_rel * heading[:, None, :]).sum(dim=-1)
    p3_along = (p3_rel * heading[:, None, :]).sum(dim=-1)
    p_mid = 0.5 * (controls[:, :, 1, :2] + controls[:, :, 2, :2])
    mid_rel = p_mid - obstacle_xy[:, None, :]
    mid_lateral = (mid_rel * left[:, None, :]).sum(dim=-1)
    crosses = torch.logical_and(p0_along < -0.02, p3_along > 0.02)
    if contact_state is not None:
        contact = torch.as_tensor(contact_state, dtype=torch.bool, device=controls.device)
        split = max(1, int(contact.shape[1]) // 2)
        lands_after = contact[:, split:, :].any(dim=1)
        swings_before = torch.logical_not(contact[:, :split, :]).any(dim=1)
        crosses = torch.logical_and(crosses, torch.logical_and(lands_after, swings_before))
    lane_abs = torch.abs(mid_lateral)
    selected_score = torch.where(crosses, lane_abs, torch.full_like(lane_abs, 1.0e6))
    arc_lane = lane + float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_lane_margin_m", 0.08))
    crossing_leg = torch.logical_and(torch.logical_and(crosses, selected_score <= arc_lane), active[:, None])
    has_leg = torch.logical_and(active, selected_score.amin(dim=1) < arc_lane)
    clearance = float(getattr(cfg.runtime, "low_small_swing_clearance_m", 0.06))
    clearance += float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_margin_m", 0.0))
    target_z = obstacle_height[:, None] + clearance
    p1_deficit = torch.relu(target_z - controls[:, :, 1, 2])
    p2_deficit = torch.relu(target_z - controls[:, :, 2, 2])
    arc_deficit = torch.maximum(p1_deficit, p2_deficit)
    # P1/P2 are Bezier control points; the curve midpoint only receives part
    # of their z motion. Back-project the required sample clearance to control
    # point space so fixed-sample foot-over metrics see the intended lift.
    arc_deficit = arc_deficit * 2.0
    arc_step = torch.clamp(
        arc_deficit,
        min=0.0,
        max=max(float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_lift_step_m", clearance)), clearance * 2.0),
    )
    arc_step = torch.where(crossing_leg, arc_step, torch.zeros_like(arc_step))
    target_lane = min(
        float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_target_lane_m", 0.05)),
        lane,
    )
    lateral_target = torch.clamp(mid_lateral, min=-target_lane, max=target_lane)
    lateral_error = lateral_target - mid_lateral
    lateral_step = torch.clamp(
        lateral_error,
        min=-float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_lateral_step_m", 0.08)),
        max=float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_lateral_step_m", 0.08)),
    )
    lateral_step = torch.where(crossing_leg, lateral_step, torch.zeros_like(lateral_step))
    lateral_delta_xy = lateral_step.unsqueeze(-1) * left[:, None, :]
    p3_lateral = (p3_rel * left[:, None, :]).sum(dim=-1)
    p3_lateral_target = torch.clamp(p3_lateral, min=-target_lane, max=target_lane)
    p3_lateral_step = torch.clamp(
        p3_lateral_target - p3_lateral,
        min=-float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_lateral_step_m", 0.08)),
        max=float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_lateral_step_m", 0.08)),
    )
    p3_lateral_step = torch.where(crossing_leg, p3_lateral_step, torch.zeros_like(p3_lateral_step))
    p3_lateral_delta_xy = p3_lateral_step.unsqueeze(-1) * left[:, None, :]
    endpoint_margin = float(getattr(cfg.runtime, "continuous_low_small_foot_over_endpoint_margin_m", 0.08))
    endpoint_deficit = torch.relu(endpoint_margin - p3_along)
    endpoint_step = torch.clamp(
        endpoint_deficit,
        min=0.0,
        max=float(getattr(cfg.runtime, "continuous_low_small_crossing_endpoint_step_m", 0.10)),
    )
    endpoint_step = torch.where(crossing_leg, endpoint_step, torch.zeros_like(endpoint_step))
    endpoint_delta_xy = endpoint_step.unsqueeze(-1) * heading[:, None, :]
    updated = controls.clone()
    updated[:, :, 1, :2] = updated[:, :, 1, :2] + lateral_delta_xy + p3_lateral_delta_xy * (1.0 / 3.0) + endpoint_delta_xy * (1.0 / 3.0)
    updated[:, :, 2, :2] = updated[:, :, 2, :2] + lateral_delta_xy + p3_lateral_delta_xy * (2.0 / 3.0) + endpoint_delta_xy * (2.0 / 3.0)
    updated[:, :, 3, :2] = updated[:, :, 3, :2] + p3_lateral_delta_xy + endpoint_delta_xy
    updated[:, :, 1, 2] = updated[:, :, 1, 2] + arc_step
    updated[:, :, 2, 2] = updated[:, :, 2, 2] + arc_step
    diagnostics = {
        # Old MPC low-small design treats foot-over as a crossing-leg diagnostic gate.
        # It must not force a touchdown/endpoint move; arc clearance is a fixed-shape
        # swing residual, while endpoint safety stays in touchdown keepout/readback.
        "qp_continuous_low_small_foot_over_update_count": zero,
        "qp_continuous_low_small_crossing_leg_count": has_leg.to(dtype=controls.dtype),
        "qp_continuous_low_small_crossing_arc_lift_count": torch.count_nonzero(
            arc_step > 1.0e-6,
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_low_small_crossing_arc_lateral_update_count": torch.count_nonzero(
            torch.logical_or(torch.abs(lateral_step) > 1.0e-6, torch.abs(p3_lateral_step) > 1.0e-6),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_low_small_crossing_endpoint_update_count": torch.count_nonzero(
            endpoint_step > 1.0e-6,
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_low_small_crossing_endpoint_deficit_max": torch.where(
            crossing_leg,
            endpoint_deficit,
            torch.zeros_like(endpoint_deficit),
        ).reshape(batch, -1).amax(dim=1).to(dtype=controls.dtype),
        "qp_continuous_low_small_crossing_arc_deficit_before_max": torch.where(
            crossing_leg,
            arc_deficit,
            torch.zeros_like(arc_deficit),
        ).reshape(batch, -1).amax(dim=1).to(dtype=controls.dtype),
        "qp_continuous_low_small_foot_over_lateral_deficit_max": torch.where(
            active,
            selected_score.amin(dim=1),
            zero,
        ).to(dtype=controls.dtype),
        "qp_continuous_low_small_foot_over_reach_reject_count": zero,
    }
    return _terrain_bound_controls(updated, terrain), diagnostics


def _reachability_update(
    controls: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    contact_state: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    sample_count = int(root_pos.shape[1])
    target = sample_controls_with_optional_gait(
        controls,
        sample_count=sample_count,
        contact_state=contact_state,
    )
    rot_world_to_body = _rpy_to_rot_matrix(root_rpy).transpose(-1, -2)
    foot_delta_w = target - root_pos.unsqueeze(2)
    foot_body = torch.einsum("btij,btkj->btki", rot_world_to_body, foot_delta_w)
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=root_pos.device, dtype=root_pos.dtype).view(1, 1, 4, 3)
    foot_hip = foot_body - hip_offsets
    reach = torch.linalg.vector_norm(foot_hip, dim=-1)
    max_reach = float(THIGH_LENGTH + CALF_LENGTH) * 0.96
    excess = torch.relu(reach - max_reach)
    per_leg_excess = excess.amax(dim=1)
    active = per_leg_excess > 1.0e-6
    if not bool(torch.any(active).item()):
        diagnostics = {
            "qp_continuous_solver_reachability_update_count": torch.zeros(
                (controls.shape[0],),
                dtype=controls.dtype,
                device=controls.device,
            ),
            "qp_continuous_solver_reachability_excess_before_max": per_leg_excess.reshape(
                controls.shape[0],
                -1,
            ).amax(dim=1).to(dtype=controls.dtype),
        }
        return controls, diagnostics

    updated = controls.clone()
    endpoint_xy = updated[:, :, 3, :2]
    anchor_xy = updated[:, :, 0, :2]
    direction = endpoint_xy - anchor_xy
    dist = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    max_step = float(getattr(cfg.runtime, "continuous_reachability_step_m", 0.16))
    shrink = torch.clamp(per_leg_excess.unsqueeze(-1) * 1.25, min=0.0, max=max_step)
    delta = -(direction / dist) * shrink
    delta = torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))
    endpoint_new = updated[:, :, 3, :2] + delta
    root_mid = root_pos[:, sample_count // 2, :].unsqueeze(1)
    anchor_z = height_at(terrain, anchor_xy).to(dtype=controls.dtype, device=controls.device)
    endpoint_z = height_at(terrain, endpoint_new).to(dtype=controls.dtype, device=controls.device)
    vertical_excess = torch.relu(torch.abs(root_mid[..., 2] - endpoint_z) - max_reach * 0.90)
    anchor_vertical_ok = torch.abs(root_mid[..., 2] - anchor_z) <= max_reach * 0.90
    snap_to_anchor = torch.logical_and(vertical_excess > 1.0e-6, anchor_vertical_ok)
    endpoint_new = torch.where(snap_to_anchor.unsqueeze(-1), anchor_xy, endpoint_new)
    delta = endpoint_new - updated[:, :, 3, :2]
    updated[:, :, 3, :2] = endpoint_new
    updated[:, :, 1, :2] = updated[:, :, 1, :2] + delta * (1.0 / 3.0)
    updated[:, :, 2, :2] = updated[:, :, 2, :2] + delta * (2.0 / 3.0)
    updated = _terrain_bound_controls(updated, terrain)
    diagnostics = {
        "qp_continuous_solver_reachability_update_count": torch.count_nonzero(
            active.reshape(active.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
        "qp_continuous_solver_reachability_excess_before_max": per_leg_excess.reshape(
            controls.shape[0],
            -1,
        ).amax(dim=1).to(dtype=controls.dtype),
        "qp_continuous_solver_reachability_delta_max": torch.linalg.vector_norm(delta, dim=-1).reshape(
            controls.shape[0],
            -1,
        ).amax(dim=1).to(dtype=controls.dtype),
        "qp_continuous_solver_reachability_anchor_snap_count": torch.count_nonzero(
            snap_to_anchor.reshape(snap_to_anchor.shape[0], -1),
            dim=1,
        ).to(dtype=controls.dtype),
    }
    return updated, diagnostics


def _body_leg_clearance_update(
    controls: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    contact_state: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    sample_count = int(root_pos.shape[1])
    target_foot = sample_controls_with_optional_gait(
        controls,
        sample_count=sample_count,
        contact_state=contact_state,
    )
    joint_angles = solve_joint_angles_from_trajectory(root_pos, root_rpy, target_foot)
    leg_points = fk_leg_points_from_joint_angles(root_pos, root_rpy, joint_angles, shank_sample_count=2)
    knee_semantic = semantic_at(terrain, leg_points.knee_pos_world[..., :2])
    knee_terrain_z = height_at(terrain, leg_points.knee_pos_world[..., :2]).to(
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    knee_deficit = torch.where(
        knee_semantic != 0,
        torch.relu(knee_terrain_z + float(cfg.runtime.body_leg_root_lift_margin_m) - leg_points.knee_pos_world[..., 2]),
        torch.zeros_like(leg_points.knee_pos_world[..., 2]),
    )
    shank_semantic = semantic_at(terrain, leg_points.shank_sample_world[..., :2])
    shank_terrain_z = height_at(terrain, leg_points.shank_sample_world[..., :2]).to(
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    shank_deficit = torch.where(
        shank_semantic != 0,
        torch.relu(
            shank_terrain_z
            + float(cfg.runtime.body_leg_root_lift_margin_m)
            - leg_points.shank_sample_world[..., 2]
        ),
        torch.zeros_like(leg_points.shank_sample_world[..., 2]),
    )
    per_leg_deficit = torch.maximum(knee_deficit, shank_deficit.amax(dim=-1))
    if contact_state is None:
        swing_mask = torch.ones_like(per_leg_deficit, dtype=torch.bool)
    else:
        swing_mask = torch.logical_not(torch.as_tensor(contact_state, dtype=torch.bool, device=controls.device))
    swing_deficit = torch.where(swing_mask, per_leg_deficit, torch.zeros_like(per_leg_deficit))
    leg_deficit = swing_deficit.amax(dim=1)
    yaw = root_rpy[:, :, 2].mean(dim=1)
    lateral_axis = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
    hip_offsets = HIP_OFFSETS_ARRAY.to(dtype=controls.dtype, device=controls.device)
    leg_side = torch.sign(hip_offsets[:, 1]).clamp(min=-1.0, max=1.0)
    lateral_dir = lateral_axis[:, None, :] * leg_side.view(1, 4, 1)
    lateral_step = (
        leg_deficit
        * float(getattr(cfg.runtime, "body_leg_xy_repair_gain", 1.0))
    ).clamp(max=float(getattr(cfg.runtime, "body_leg_xy_repair_step_m", 0.03)))
    lateral_delta = lateral_step.unsqueeze(-1) * lateral_dir
    updated_controls = controls.clone()
    updated_controls[:, :, 1, :2] = updated_controls[:, :, 1, :2] + lateral_delta * (1.0 / 3.0)
    updated_controls[:, :, 2, :2] = updated_controls[:, :, 2, :2] + lateral_delta * (2.0 / 3.0)
    underbody = _underbody_points(root_pos)
    underbody_semantic = semantic_at(terrain, underbody[..., :2])
    underbody_terrain_z = height_at(terrain, underbody[..., :2]).to(dtype=root_pos.dtype, device=root_pos.device)
    underbody_deficit = torch.where(
        underbody_semantic != 0,
        torch.relu(underbody_terrain_z + 0.015 - underbody[..., 2]),
        torch.zeros_like(underbody[..., 2]),
    )
    frame_underbody_deficit = underbody_deficit.amax(dim=-1)
    frame_deficit = per_leg_deficit.amax(dim=-1).clamp(max=float(cfg.runtime.body_leg_root_lift_max_m))
    frame_deficit = torch.maximum(
        frame_deficit,
        frame_underbody_deficit.clamp(max=float(cfg.runtime.body_leg_root_lift_max_m)),
    )
    if int(frame_deficit.shape[1]) > 1:
        left = torch.nn.functional.pad(frame_deficit[:, :-1], (1, 0))
        right = torch.nn.functional.pad(frame_deficit[:, 1:], (0, 1))
        frame_deficit = torch.maximum(frame_deficit, torch.maximum(left, right) * 0.5)
    updated_root = root_pos.clone()
    updated_root[..., 2] = updated_root[..., 2] + frame_deficit
    diagnostics = {
        "qp_continuous_solver_body_leg_clearance_update_count": torch.count_nonzero(
            frame_deficit > 1.0e-6,
            dim=1,
        ).to(dtype=root_pos.dtype),
        "qp_continuous_solver_body_leg_lateral_update_count": torch.count_nonzero(
            lateral_step > 1.0e-6,
            dim=1,
        ).to(dtype=root_pos.dtype),
        "qp_continuous_solver_body_leg_clearance_deficit_before_max": per_leg_deficit.reshape(
            per_leg_deficit.shape[0],
            -1,
        ).amax(dim=1),
    }
    return _terrain_bound_controls(updated_controls, terrain), updated_root, diagnostics


def continuous_qp_update(
    controls: ContinuousTrajectoryControls,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    *,
    command: Tensor | None = None,
    root_pos: Tensor | None = None,
    root_rpy: Tensor | None = None,
    contact_state: Tensor | None = None,
) -> tuple[ContinuousTrajectoryControls, dict[str, Tensor]]:
    """Apply one fixed-shape loss-driven update to Bezier controls.

    This is the first continuous-QP scaffold: it reduces bad touchdown
    foothold height variation by evaluating a fixed +/-xy stencil and taking
    one bounded step toward the lowest fixed-shape loss. It does not construct
    per-obstacle constraints or perform post-hoc trajectory repair.
    """

    return coupled_qp_update(
        controls,
        terrain,
        cfg,
        command=command,
        contact_state=contact_state,
    )

    current = _terrain_bound_controls(torch.as_tensor(controls.foot_control_w), terrain)
    root_pos_control = torch.as_tensor(controls.root_pos_w)
    root_rpy_control = torch.as_tensor(controls.root_rpy)
    updated = current.clone()
    touchdown_xy = current[:, :, 3, :2]
    touchdown_xy_updated, touchdown_diagnostics = _touchdown_field_qp_step(touchdown_xy, terrain, cfg)
    root_pos_updated, root_diagnostics = _root_terrain_progress_update(root_pos_control, terrain, cfg)
    root_rpy_control, root_attitude_diagnostics = _root_attitude_level_update(root_rpy_control, terrain, cfg)
    root_pos_updated, updated, low_small_progress_diagnostics = _low_small_crossing_progress_update(
        root_pos_updated,
        updated,
        terrain,
        command,
        cfg,
    )
    updated, low_small_foot_over_diagnostics = _low_small_foot_over_update(
        updated,
        root_pos_updated,
        root_rpy_control,
        terrain,
        command,
        cfg,
        contact_state=contact_state,
    )
    updated, root_pos_updated, body_leg_diagnostics = _body_leg_clearance_update(
        updated,
        root_pos_updated,
        root_rpy_control,
        terrain,
        cfg,
        contact_state=contact_state,
    )
    root_progress_scale = root_diagnostics["qp_continuous_root_progress_scale_min"].to(
        dtype=updated.dtype,
        device=updated.device,
    )
    foot_start_xy = updated[:, :, :1, :2]
    updated[:, :, :, :2] = foot_start_xy + (updated[:, :, :, :2] - foot_start_xy) * root_progress_scale.view(-1, 1, 1, 1)
    delta = touchdown_xy_updated - touchdown_xy
    updated[:, :, 3, :2] = touchdown_xy_updated
    updated[:, :, 1, :2] = updated[:, :, 1, :2] + delta * (1.0 / 3.0)
    updated[:, :, 2, :2] = updated[:, :, 2, :2] + delta * (2.0 / 3.0)
    updated = _terrain_bound_controls(updated, terrain)
    updated, reachability_diagnostics = _reachability_update(
        updated,
        root_pos_updated,
        root_rpy_control,
        terrain,
        cfg,
        contact_state=contact_state,
    )
    updated, joint_limit_diagnostics = _joint_limit_readback_update(
        updated,
        root_pos_updated,
        root_rpy_control,
        terrain,
        cfg,
        contact_state=contact_state,
    )
    readback_diagnostics: dict[str, Tensor] = {}
    readback_root_pos = root_pos_updated
    if root_pos is not None:
        readback_root_pos = root_pos_updated.to(dtype=root_pos.dtype, device=root_pos.device)
    readback_root_rpy = root_rpy_control
    if root_rpy is not None:
        readback_root_rpy = root_rpy_control.to(dtype=root_rpy.dtype, device=root_rpy.device)
    if readback_root_pos is not None and readback_root_rpy is not None:
        updated, root_pos_updated, readback_diagnostics = _fk_readback_update(
            updated,
            readback_root_pos,
            readback_root_rpy,
            terrain,
            cfg,
            contact_state=contact_state,
        )
        readback_root_pos = root_pos_updated
    updated, terrain_clearance_diagnostics = _terrain_clearance_lift(updated, terrain, cfg, contact_state=contact_state)
    updated = _terrain_bound_controls(updated, terrain)
    updated, swing_diagnostics = _swing_clearance_lift(updated, terrain, cfg)
    updated = _terrain_bound_controls(updated, terrain)
    final_readback_diagnostics: dict[str, Tensor] = {}
    final_swing_diagnostics: dict[str, Tensor] = {}
    final_low_small_foot_over_diagnostics: dict[str, Tensor] = {}
    final_body_leg_diagnostics: dict[str, Tensor] = {}
    if readback_root_pos is not None and readback_root_rpy is not None:
        updated, root_pos_updated, final_readback_diagnostics = _fk_readback_update(
            updated,
            readback_root_pos,
            readback_root_rpy,
            terrain,
            cfg,
            contact_state=contact_state,
        )
        readback_root_pos = root_pos_updated
        updated = _terrain_bound_controls(updated, terrain)
        updated, final_low_small_foot_over_diagnostics = _low_small_foot_over_update(
            updated,
            readback_root_pos,
            readback_root_rpy,
            terrain,
            command,
            cfg,
            contact_state=contact_state,
        )
        updated = _terrain_bound_controls(updated, terrain)
        updated, final_terrain_clearance_diagnostics = _terrain_clearance_lift(
            updated,
            terrain,
            cfg,
            contact_state=contact_state,
        )
        updated = _terrain_bound_controls(updated, terrain)
        updated, final_swing_diagnostics = _swing_clearance_lift(updated, terrain, cfg)
        updated = _terrain_bound_controls(updated, terrain)
        updated, root_pos_updated, final_readback_after_crossing = _fk_readback_update(
            updated,
            readback_root_pos,
            readback_root_rpy,
            terrain,
            cfg,
            contact_state=contact_state,
        )
        readback_root_pos = root_pos_updated
        updated = _terrain_bound_controls(updated, terrain)
        updated, root_pos_updated, final_body_leg_diagnostics = _body_leg_clearance_update(
            updated,
            root_pos_updated,
            readback_root_rpy,
            terrain,
            cfg,
            contact_state=contact_state,
        )
        readback_root_pos = root_pos_updated
        extra_readback_passes = int(
            int(getattr(cfg.runtime, "continuous_low_small_final_readback_passes", 0))
            if torch.any(
                final_low_small_foot_over_diagnostics.get(
                    "qp_continuous_low_small_crossing_leg_count",
                    torch.zeros((updated.shape[0],), dtype=updated.dtype, device=updated.device),
                )
                > 0.0
            ).item()
            else 0
        )
        for _ in range(extra_readback_passes):
            updated, root_pos_updated, final_readback_after_body_leg = _fk_readback_update(
                updated,
                readback_root_pos,
                readback_root_rpy,
                terrain,
                cfg,
                contact_state=contact_state,
            )
            readback_root_pos = root_pos_updated
            updated = _terrain_bound_controls(updated, terrain)
            for name, value in final_readback_after_body_leg.items():
                if name.endswith("_count"):
                    final_readback_diagnostics[name] = final_readback_diagnostics.get(name, torch.zeros_like(value)) + value
                elif name.endswith("_before_max") or name.endswith("_max"):
                    final_readback_diagnostics[name] = torch.maximum(final_readback_diagnostics.get(name, value), value)
                else:
                    final_readback_diagnostics[name] = value
        for name, value in final_readback_after_crossing.items():
            if name.endswith("_count"):
                final_readback_diagnostics[name] = final_readback_diagnostics.get(name, torch.zeros_like(value)) + value
            elif name.endswith("_before_max") or name.endswith("_max"):
                final_readback_diagnostics[name] = torch.maximum(final_readback_diagnostics.get(name, value), value)
            else:
                final_readback_diagnostics[name] = value
    diagnostics = dict(touchdown_diagnostics)
    diagnostics.update(root_diagnostics)
    diagnostics.update(root_attitude_diagnostics)
    diagnostics.update(low_small_progress_diagnostics)
    diagnostics.update(low_small_foot_over_diagnostics)
    diagnostics.update(body_leg_diagnostics)
    diagnostics.update(reachability_diagnostics)
    diagnostics.update(joint_limit_diagnostics)
    diagnostics.update(terrain_clearance_diagnostics)
    diagnostics.update(swing_diagnostics)
    diagnostics.update(readback_diagnostics)
    if final_terrain_clearance_diagnostics:
        for name, value in final_terrain_clearance_diagnostics.items():
            if name.endswith("_count"):
                diagnostics[name] = diagnostics.get(name, torch.zeros_like(value)) + value
            elif name.endswith("_max") or name.endswith("_before_max"):
                diagnostics[name] = torch.maximum(diagnostics.get(name, value), value)
            else:
                diagnostics[name] = value
    if final_swing_diagnostics:
        for name, value in final_swing_diagnostics.items():
            if name.endswith("_count"):
                diagnostics[name] = diagnostics.get(name, torch.zeros_like(value)) + value
            elif name.endswith("_max") or name.endswith("_before_max"):
                diagnostics[name] = torch.maximum(diagnostics.get(name, value), value)
            else:
                diagnostics[name] = value
    for name, value in final_body_leg_diagnostics.items():
        if name.endswith("_count"):
            diagnostics[name] = diagnostics.get(name, torch.zeros_like(value)) + value
        elif name.endswith("_max") or name.endswith("_before_max"):
            diagnostics[name] = torch.maximum(diagnostics.get(name, value), value)
        else:
            diagnostics[name] = value
    for name, value in final_low_small_foot_over_diagnostics.items():
        if name.endswith("_count"):
            diagnostics[name] = diagnostics.get(name, torch.zeros_like(value)) + value
        elif name.endswith("_max"):
            diagnostics[name] = torch.maximum(diagnostics.get(name, value), value)
        else:
            diagnostics[name] = value
    for name, value in final_readback_diagnostics.items():
        if name.endswith("_count"):
            diagnostics[name] = diagnostics.get(name, torch.zeros_like(value)) + value
        elif name.endswith("_before_max") or name.endswith("_max"):
            diagnostics[name] = torch.maximum(diagnostics.get(name, value), value)
        else:
            diagnostics[name] = value
    return ContinuousTrajectoryControls(
        foot_control_w=updated,
        root_pos_w=root_pos_updated,
        root_rpy=root_rpy_control,
    ), diagnostics


__all__ = ["continuous_qp_update"]
