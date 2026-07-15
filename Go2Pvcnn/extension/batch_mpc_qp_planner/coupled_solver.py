"""Coupled fixed-shape trajectory optimizer for the MPC-QP backend."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.batch_mpc_planner.kinematics import (
    HIP_OFFSETS_ARRAY,
    CALF_LENGTH,
    THIGH_LENGTH,
    _rpy_to_rot_matrix,
    fk_feet_from_joint_angles,
    fk_leg_points_from_joint_angles,
    solve_joint_angles_from_trajectory,
)
from extension.batch_mpc_planner.terrain import height_at, semantic_at
from extension.batch_mpc_planner.types import MpcPlannerTerrain

from .config import MpcQpPlannerCfg
from .continuous import ContinuousTrajectoryControls, sample_controls_with_optional_gait
from .fields import build_qp_fields
from .losses import _footprint_offsets


def _underbody_points(root_pos: Tensor) -> Tensor:
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
    return root_pos[:, :, None, :] + offsets.view(1, 1, 5, 3)


def _terrain_bound_controls(controls: Tensor, terrain: MpcPlannerTerrain) -> Tensor:
    out = controls.clone()
    touchdown_z = height_at(terrain, out[:, :, 3, :2]).to(dtype=out.dtype, device=out.device)
    out[:, :, 3, 2] = touchdown_z
    return out


def _row_max(value: Tensor, batch: int) -> Tensor:
    return value.reshape(batch, -1).amax(dim=1)


def _row_count(mask: Tensor, batch: int, *, dtype: torch.dtype) -> Tensor:
    return torch.count_nonzero(mask.reshape(batch, -1), dim=1).to(dtype=dtype)


def _row_min(value: Tensor, batch: int) -> Tensor:
    return value.reshape(batch, -1).amin(dim=1)


def _cmd_axes(command: Tensor | None, batch: int, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    if command is None:
        heading = torch.zeros((batch, 2), dtype=dtype, device=device)
        heading[:, 0] = 1.0
        active = torch.zeros((batch,), dtype=torch.bool, device=device)
    else:
        cmd = torch.as_tensor(command, dtype=dtype, device=device)
        if cmd.ndim == 1:
            cmd = cmd.view(1, -1).expand(batch, -1)
        heading = cmd[:, :2]
        speed = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
        active = speed.squeeze(-1) > 1.0e-5
        heading = heading / speed.clamp_min(1.0e-6)
        heading = torch.where(active[:, None], heading, torch.tensor((1.0, 0.0), dtype=dtype, device=device))
    left = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    return heading, left, active


def _low_small_target(
    controls: Tensor,
    root_pos: Tensor,
    terrain: MpcPlannerTerrain,
    command: Tensor | None,
    cfg: MpcQpPlannerCfg,
    *,
    anchor_root_xy: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch = int(controls.shape[0])
    dtype = controls.dtype
    device = controls.device
    heading, left, active_cmd = _cmd_axes(command, batch, dtype=dtype, device=device)
    sample_count = max(3, int(getattr(cfg.runtime, "continuous_low_small_progress_sample_count", 17)))
    lookahead = float(getattr(cfg.runtime, "continuous_low_small_progress_lookahead_m", 0.48))
    lane = float(getattr(cfg.runtime, "continuous_low_small_progress_lane_half_width_m", 0.14))
    root_xy = root_pos[:, 0, :2] if anchor_root_xy is None else anchor_root_xy.to(dtype=dtype, device=device)
    along_samples = torch.linspace(0.0, lookahead, sample_count, dtype=dtype, device=device)
    lateral_samples = torch.tensor((-lane, 0.0, lane), dtype=dtype, device=device)
    points = (
        root_xy[:, None, None, :]
        + along_samples.view(1, sample_count, 1, 1) * heading[:, None, None, :]
        + lateral_samples.view(1, 1, 3, 1) * left[:, None, None, :]
    )
    semantic = semantic_at(terrain, points.reshape(batch, sample_count * 3, 2)).reshape(batch, sample_count, 3)
    height = height_at(terrain, points.reshape(batch, sample_count * 3, 2)).reshape(batch, sample_count, 3).to(
        dtype=dtype,
        device=device,
    )
    root_ground = height_at(terrain, root_xy[:, None, :]).reshape(batch, 1, 1).to(dtype=dtype, device=device)
    low_small_height = float(getattr(cfg.losses.low_small_crossing, "high_small_relative_height_m", 0.30))
    low_small = torch.logical_and(semantic == 1, (height - root_ground) <= low_small_height)
    active = torch.logical_and(low_small.any(dim=(1, 2)), active_cmd)
    masked_along = torch.where(
        low_small,
        along_samples.view(1, sample_count, 1).expand(batch, -1, 3),
        torch.full((batch, sample_count, 3), lookahead + 1.0, dtype=dtype, device=device),
    )
    obstacle_along = masked_along.reshape(batch, -1).amin(dim=1)
    obstacle_xy = root_xy + obstacle_along[:, None] * heading
    obstacle_h = torch.where(low_small, height, root_ground.expand_as(height)).reshape(batch, -1).amax(dim=1)
    local_r = float(getattr(cfg.runtime, "continuous_low_small_progress_lane_half_width_m", 0.14))
    local_offsets = torch.tensor(
        (
            (0.0, 0.0),
            (local_r, 0.0),
            (-local_r, 0.0),
            (0.0, local_r),
            (0.0, -local_r),
            (local_r, local_r),
            (local_r, -local_r),
            (-local_r, local_r),
            (-local_r, -local_r),
        ),
        dtype=dtype,
        device=device,
    )
    local_xy = obstacle_xy[:, None, :] + local_offsets.view(1, -1, 2)
    local_h = height_at(terrain, local_xy).to(dtype=dtype, device=device)
    obstacle_h = torch.maximum(obstacle_h, local_h.amax(dim=1))
    return obstacle_xy, obstacle_h, heading, left, active


def _root_terrain_risk(
    root_pos: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = int(root_pos.shape[0])
    sample_count = max(2, int(getattr(cfg.runtime, "terrain_step_cap_sample_count", 5)))
    idx = torch.linspace(0, root_pos.shape[1] - 1, sample_count, dtype=torch.float32, device=root_pos.device)
    idx = idx.round().to(dtype=torch.long)
    sampled_xy = root_pos.index_select(1, idx)[..., :2]
    sampled_h = height_at(terrain, sampled_xy.reshape(batch, -1, 2)).reshape(batch, sample_count).to(
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    sampled_sem = semantic_at(terrain, sampled_xy.reshape(batch, -1, 2)).reshape(batch, sample_count)
    baseline_h = sampled_h.amin(dim=1, keepdim=True)
    low_small_height = float(getattr(cfg.losses.low_small_crossing, "high_small_relative_height_m", 0.30))
    low_small = torch.logical_and(sampled_sem == 1, (sampled_h - baseline_h) <= low_small_height)
    risk_h = torch.where(low_small, baseline_h.expand_as(sampled_h), sampled_h)
    variation = risk_h.amax(dim=1) - risk_h.amin(dim=1)
    threshold = float(getattr(cfg.runtime, "terrain_height_variation_threshold_m", 0.06))
    risky = variation > threshold
    progress = torch.linalg.vector_norm(root_pos[:, -1, :2] - root_pos[:, 0, :2], dim=-1)
    risk_loss = torch.where(risky, progress.square() + torch.relu(variation - threshold).square() * 10.0, torch.zeros_like(progress))
    over = torch.clamp((variation - threshold) / max(threshold, 1.0e-6), min=0.0, max=1.0)
    min_scale = float(getattr(cfg.runtime, "terrain_step_cap_min_scale", 0.55))
    scale = torch.where(
        risky,
        1.0 - (1.0 - min_scale) * over.to(dtype=root_pos.dtype, device=root_pos.device),
        torch.ones_like(variation, dtype=root_pos.dtype, device=root_pos.device),
    )
    return risk_loss, risky.to(dtype=root_pos.dtype, device=root_pos.device), scale


def _loss_terms(
    controls: Tensor,
    root_pos: Tensor,
    root_rpy: Tensor,
    nominal_controls: Tensor,
    nominal_root: Tensor,
    terrain: MpcPlannerTerrain,
    command: Tensor | None,
    cfg: MpcQpPlannerCfg,
    contact_state: Tensor | None,
) -> tuple[Tensor, dict[str, Tensor]]:
    batch = int(controls.shape[0])
    controls = _terrain_bound_controls(controls, terrain)
    foot = sample_controls_with_optional_gait(controls, sample_count=int(root_pos.shape[1]), contact_state=contact_state)
    joint = solve_joint_angles_from_trajectory(root_pos, root_rpy, foot)
    fk_foot = fk_feet_from_joint_angles(root_pos, root_rpy, joint)
    raw_joint = solve_joint_angles_from_trajectory(root_pos, root_rpy, foot, clamp_to_limits=False).reshape(batch, int(root_pos.shape[1]), 4, 3)
    limits = torch.tensor(
        ((-1.0472, 1.0472), (-1.5708, 3.4907), (-2.7227, -0.8378)),
        dtype=controls.dtype,
        device=controls.device,
    ).view(1, 1, 1, 3, 2)
    joint_violation = torch.maximum(torch.relu(limits[..., 0] - raw_joint), torch.relu(raw_joint - limits[..., 1])).amax(dim=-1)

    fields = build_qp_fields(terrain, eps_m=float(getattr(cfg.runtime, "continuous_foothold_probe_m", 0.04)))
    touchdown_xy = controls[:, :, 3, :2]
    td_field = fields.query(touchdown_xy)
    footprint_offsets = _footprint_offsets(
        radius_m=float(getattr(cfg.runtime, "continuous_footprint_radius_m", 0.04)),
        dtype=controls.dtype,
        device=controls.device,
    )
    footprint_xy = touchdown_xy[..., None, :] + footprint_offsets.view(1, 1, -1, 2)
    footprint_sem = fields.query(footprint_xy.reshape(batch, -1, 2)).semantic_risk.reshape(batch, 4, -1)
    rough_target = float(getattr(cfg.runtime, "continuous_foothold_variation_target_m", 0.03))
    touchdown_sem_violation = torch.relu(td_field.semantic_risk)
    touchdown_rough_violation = torch.relu(td_field.roughness - rough_target)
    touchdown_sem = touchdown_sem_violation.square().mean(dim=1) + footprint_sem.square().mean(dim=(1, 2))
    touchdown_quality = touchdown_rough_violation.square().mean(dim=1)

    terrain_z = height_at(terrain, foot[..., :2].reshape(batch, -1, 2)).reshape(*foot.shape[:-1]).to(
        dtype=foot.dtype,
        device=foot.device,
    )
    foot_sem = semantic_at(terrain, foot[..., :2].reshape(batch, -1, 2)).reshape(*foot.shape[:-1])
    swing = torch.ones_like(foot_sem, dtype=torch.bool) if contact_state is None else torch.logical_not(contact_state)
    terrain_margin = float(getattr(cfg.runtime, "continuous_terrain_clearance_m", 0.018))
    swing_clearance = torch.relu(terrain_z + terrain_margin - foot[..., 2])
    low_small_clearance = torch.relu(
        terrain_z
        + float(getattr(cfg.runtime, "low_small_swing_clearance_m", 0.06))
        - foot[..., 2]
    )
    terrain_clearance_loss = torch.where(swing, swing_clearance.square(), torch.zeros_like(swing_clearance)).mean(dim=(1, 2))
    swing_loss = terrain_clearance_loss
    swing_loss = swing_loss + torch.where(
        torch.logical_and(swing, foot_sem == 1),
        low_small_clearance.square(),
        torch.zeros_like(low_small_clearance),
    ).mean(dim=(1, 2)) * 80.0

    obstacle_xy, obstacle_h, heading, left, active_low = _low_small_target(
        controls,
        root_pos,
        terrain,
        command,
        cfg,
        anchor_root_xy=nominal_root[:, 0, :2].detach(),
    )
    mid_xy = 0.5 * (controls[:, :, 1, :2] + controls[:, :, 2, :2])
    mid_rel = mid_xy - obstacle_xy[:, None, :]
    mid_along = (mid_rel * heading[:, None, :]).sum(dim=-1)
    mid_lat = (mid_rel * left[:, None, :]).sum(dim=-1)
    p0_along = ((controls[:, :, 0, :2] - obstacle_xy[:, None, :]) * heading[:, None, :]).sum(dim=-1)
    p3_along = ((controls[:, :, 3, :2] - obstacle_xy[:, None, :]) * heading[:, None, :]).sum(dim=-1)
    crosses = torch.logical_and(p0_along < -0.02, p3_along > 0.02)
    crossing_active = torch.logical_and(crosses, active_low[:, None])
    clearance_req = float(getattr(cfg.runtime, "low_small_swing_clearance_m", 0.06))
    crossing_z_req = obstacle_h[:, None] + clearance_req + float(
        getattr(cfg.runtime, "continuous_low_small_crossing_arc_margin_m", 0.04)
    )
    p1p2_z = controls[:, :, 2, 2]
    crossing_height = torch.where(crossing_active, torch.relu(crossing_z_req - p1p2_z).square(), torch.zeros_like(p1p2_z))
    target_lane = float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_target_lane_m", 0.03))
    crossing_lat = torch.where(crossing_active, torch.relu(torch.abs(mid_lat) - target_lane).square(), torch.zeros_like(mid_lat))
    crossing_mid_along = torch.where(
        crossing_active,
        torch.relu(torch.abs(mid_along) - 0.035).square(),
        torch.zeros_like(mid_along),
    )
    crossing_loss = (crossing_height + crossing_lat + crossing_mid_along * 2.0).mean(dim=1)
    lane_half = float(getattr(cfg.runtime, "continuous_low_small_progress_lane_half_width_m", 0.14))
    if contact_state is None:
        swing_any = torch.ones_like(p1p2_z, dtype=torch.bool)
    else:
        contact_bool = torch.as_tensor(contact_state, dtype=torch.bool, device=controls.device)
        swing_any = torch.logical_not(contact_bool).any(dim=1)
    front_swing = p0_along < -0.02
    arc_active = torch.logical_and(active_low[:, None], torch.logical_and(swing_any, front_swing))
    arc_height_loss = torch.where(
        arc_active,
        torch.relu(obstacle_h[:, None] + clearance_req + 0.035 - p1p2_z).square(),
        torch.zeros_like(p1p2_z),
    ).mean(dim=1) * 520.0
    arc_lane_loss = torch.where(
        arc_active,
        torch.relu(torch.abs(mid_lat) - lane_half * 0.45).square(),
        torch.zeros_like(mid_lat),
    ).mean(dim=1) * 180.0
    arc_along_loss = torch.where(
        arc_active,
        torch.relu(torch.abs(mid_along) - 0.035).square(),
        torch.zeros_like(mid_along),
    ).mean(dim=1) * 260.0
    current_end_along = ((root_pos[:, -1, :2] - root_pos[:, 0, :2]) * heading).sum(dim=-1)
    obstacle_along = ((obstacle_xy - root_pos[:, 0, :2]) * heading).sum(dim=-1)
    progress_margin = float(getattr(cfg.runtime, "continuous_low_small_progress_margin_m", 0.06))
    progress_deficit = torch.where(
        active_low,
        torch.relu(obstacle_along + progress_margin - current_end_along),
        torch.zeros_like(current_end_along),
    )
    root_progress_loss = progress_deficit.square() * 120.0
    root_mid_idx = int(root_pos.shape[1]) // 2
    root_mid_ground = height_at(terrain, root_pos[:, root_mid_idx : root_mid_idx + 1, :2]).reshape(batch).to(
        dtype=root_pos.dtype,
        device=root_pos.device,
    )
    root_crossing_target = obstacle_h + 0.34
    root_crossing_lift = torch.where(
        active_low,
        torch.relu(torch.maximum(root_crossing_target, root_mid_ground + 0.30) - root_pos[:, root_mid_idx, 2]).square(),
        torch.zeros_like(obstacle_h),
    ) * 90.0
    endpoint_deficit = torch.where(
        active_low[:, None],
        torch.relu(obstacle_along[:, None] + 0.08 - p3_along),
        torch.zeros_like(p3_along),
    )
    p3_lat = ((controls[:, :, 3, :2] - obstacle_xy[:, None, :]) * left[:, None, :]).sum(dim=-1)
    endpoint_lane_loss = torch.where(
        active_low[:, None],
        torch.relu(torch.abs(p3_lat) - float(getattr(cfg.runtime, "continuous_low_small_progress_lane_half_width_m", 0.14)) * 0.75).square(),
        torch.zeros_like(p3_lat),
    ).mean(dim=1) * 120.0
    endpoint_progress_loss = endpoint_deficit.square().mean(dim=1) * 260.0 + endpoint_lane_loss
    sample_rel = foot[..., :2] - obstacle_xy[:, None, None, :]
    sample_along = (sample_rel * heading[:, None, None, :]).sum(dim=-1)
    sample_lat = (sample_rel * left[:, None, None, :]).sum(dim=-1)
    sample_lane = torch.abs(sample_lat) <= lane_half
    sample_near = torch.abs(sample_along) <= 0.14
    sample_crossing = torch.logical_and(torch.logical_and(swing, sample_lane), torch.logical_and(sample_near, active_low[:, None, None]))
    sample_req_z = obstacle_h[:, None, None] + float(getattr(cfg.runtime, "low_small_swing_clearance_m", 0.06)) + 0.025
    sample_crossing_loss = torch.where(
        sample_crossing,
        torch.relu(sample_req_z - foot[..., 2]).square(),
        torch.zeros_like(foot[..., 2]),
    ).mean(dim=(1, 2)) * 420.0
    leg_active = arc_active.to(dtype=foot.dtype, device=foot.device)
    swing_weight = swing.to(dtype=foot.dtype, device=foot.device)
    sample_dist = sample_along.square() + (sample_lat / max(lane_half, 1.0e-6)).square() * 0.012
    logits = -sample_dist / 0.012 + torch.log(swing_weight.clamp_min(1.0e-6))
    nearest_weight = torch.softmax(logits, dim=1) * leg_active[:, None, :]
    persistent_sample_height = (nearest_weight * torch.relu(sample_req_z - foot[..., 2]).square()).sum(dim=(1, 2))
    persistent_sample_lane = (nearest_weight * (sample_along.square() + sample_lat.square() * 0.35)).sum(dim=(1, 2))
    fk_sample_rel = fk_foot[..., :2] - obstacle_xy[:, None, None, :]
    fk_sample_along = (fk_sample_rel * heading[:, None, None, :]).sum(dim=-1)
    fk_sample_lat = (fk_sample_rel * left[:, None, None, :]).sum(dim=-1)
    fk_sample_dist = fk_sample_along.square() + (fk_sample_lat / max(lane_half, 1.0e-6)).square() * 0.012
    fk_logits = -fk_sample_dist / 0.012 + torch.log(swing_weight.clamp_min(1.0e-6))
    fk_nearest_weight = torch.softmax(fk_logits, dim=1) * leg_active[:, None, :]
    fk_persistent_height = (fk_nearest_weight * torch.relu(sample_req_z - fk_foot[..., 2]).square()).sum(dim=(1, 2))
    fk_persistent_lane = (
        fk_nearest_weight * (fk_sample_along.square() + fk_sample_lat.square() * 0.35)
    ).sum(dim=(1, 2))
    sample_crossing_loss = (
        sample_crossing_loss
        + persistent_sample_height * 680.0
        + persistent_sample_lane * 260.0
        + fk_persistent_height * 900.0
        + fk_persistent_lane * 320.0
    )

    readback_vec = fk_foot - foot
    readback = torch.linalg.vector_norm(readback_vec, dim=-1)
    rot = _rpy_to_rot_matrix(root_rpy.reshape(-1, 3)).reshape(batch, int(root_pos.shape[1]), 3, 3)
    hip_offsets = HIP_OFFSETS_ARRAY.to(dtype=foot.dtype, device=foot.device).view(1, 1, 4, 3)
    hip_world = root_pos[:, :, None, :] + torch.matmul(rot[:, :, None, :, :], hip_offsets[..., None]).squeeze(-1)
    foot_to_hip = torch.linalg.vector_norm(foot - hip_world, dim=-1)
    reach_max = float(THIGH_LENGTH + CALF_LENGTH - 0.02)
    reach_min = 0.10
    reach_violation = torch.relu(foot_to_hip - reach_max) + torch.relu(reach_min - foot_to_hip)
    reach_loss = (
        readback.square().mean(dim=(1, 2)) * 260.0
        + joint_violation.square().mean(dim=(1, 2)) * 180.0
        + reach_violation.square().mean(dim=(1, 2)) * 360.0
    )

    leg_points = fk_leg_points_from_joint_angles(root_pos, root_rpy, joint, shank_sample_count=2)
    knee_h = height_at(terrain, leg_points.knee_pos_world[..., :2].reshape(batch, -1, 2)).reshape(*leg_points.knee_pos_world.shape[:-1]).to(dtype=controls.dtype, device=controls.device)
    knee_sem = semantic_at(terrain, leg_points.knee_pos_world[..., :2].reshape(batch, -1, 2)).reshape(*leg_points.knee_pos_world.shape[:-1])
    shank_h = height_at(terrain, leg_points.shank_sample_world[..., :2].reshape(batch, -1, 2)).reshape(*leg_points.shank_sample_world.shape[:-1]).to(dtype=controls.dtype, device=controls.device)
    shank_sem = semantic_at(terrain, leg_points.shank_sample_world[..., :2].reshape(batch, -1, 2)).reshape(*leg_points.shank_sample_world.shape[:-1])
    margin = float(getattr(cfg.runtime, "body_leg_root_lift_margin_m", 0.08))
    knee_bad = torch.where(knee_sem != 0, torch.relu(knee_h + margin - leg_points.knee_pos_world[..., 2]).square(), torch.zeros_like(knee_h))
    shank_bad = torch.where(shank_sem != 0, torch.relu(shank_h + margin - leg_points.shank_sample_world[..., 2]).square(), torch.zeros_like(shank_h))
    underbody = _underbody_points(root_pos)
    under_h = height_at(terrain, underbody[..., :2].reshape(batch, -1, 2)).reshape(*underbody.shape[:-1]).to(dtype=controls.dtype, device=controls.device)
    under_sem = semantic_at(terrain, underbody[..., :2].reshape(batch, -1, 2)).reshape(*underbody.shape[:-1])
    under_bad = torch.where(under_sem != 0, torch.relu(under_h + 0.015 - underbody[..., 2]).square(), torch.zeros_like(under_h))
    body_loss = knee_bad.mean(dim=(1, 2)) * 40.0 + shank_bad.mean(dim=(1, 2, 3)) * 40.0 + under_bad.mean(dim=(1, 2)) * 80.0

    root_ground = height_at(terrain, root_pos[..., :2].reshape(batch, -1, 2)).reshape(batch, -1).to(dtype=controls.dtype, device=controls.device)
    root_low = float(getattr(cfg.runtime, "continuous_plane_root_height_min_m", 0.30))
    root_high = float(getattr(cfg.runtime, "continuous_plane_root_height_max_m", 0.36))
    root_clearance = root_pos[..., 2] - root_ground
    root_high_weight = 2.0 if float(getattr(cfg.runtime, "continuous_fk_root_z_gain", 0.85)) > 0.0 else 0.0
    root_loss = torch.relu(root_low - root_clearance).square().mean(dim=1) * 30.0 + torch.relu(root_clearance - root_high).square().mean(dim=1) * root_high_weight
    root_z_first = root_pos[:, 1:, 2] - root_pos[:, :-1, 2]
    root_z_jump_loss = (
        torch.relu(torch.abs(root_z_first) - 0.035).square().mean(dim=1)
        if int(root_z_first.shape[1])
        else torch.zeros((batch,), dtype=controls.dtype, device=controls.device)
    )
    attitude_loss = root_rpy[..., :2].square().mean(dim=(1, 2)) * 5.0
    root_risk_loss, root_risky, root_progress_scale = _root_terrain_risk(root_pos, terrain, cfg)

    first = foot[:, 1:] - foot[:, :-1]
    second = first[:, 1:] - first[:, :-1]
    smooth = second.square().mean(dim=(1, 2, 3)) if int(second.shape[1]) else torch.zeros((batch,), dtype=controls.dtype, device=controls.device)
    foot_frame_jump = torch.linalg.vector_norm(first, dim=-1) if int(first.shape[1]) else torch.zeros((batch, 0, 4), dtype=controls.dtype, device=controls.device)
    foot_jump_target = float(getattr(cfg.runtime, "continuous_foot_frame_jump_target_m", 0.18))
    foot_jump_loss = (
        torch.relu(foot_frame_jump - foot_jump_target).square().mean(dim=(1, 2))
        if int(first.shape[1])
        else torch.zeros((batch,), dtype=controls.dtype, device=controls.device)
    )
    joint_first = joint[:, 1:] - joint[:, :-1]
    joint_second = joint_first[:, 1:] - joint_first[:, :-1]
    joint_smooth = (
        joint_second.square().mean(dim=(1, 2))
        if int(joint_second.shape[1])
        else torch.zeros((batch,), dtype=controls.dtype, device=controls.device)
    )
    joint_frame_jump = (
        torch.linalg.vector_norm(joint_first.reshape(batch, int(joint_first.shape[1]), 4, 3), dim=-1)
        if int(joint_first.shape[1])
        else torch.zeros((batch, 0, 4), dtype=controls.dtype, device=controls.device)
    )
    joint_jump_target = float(getattr(cfg.runtime, "continuous_joint_frame_jump_target_rad", 0.55))
    joint_jump_loss = (
        torch.relu(joint_frame_jump - joint_jump_target).square().mean(dim=(1, 2))
        if int(joint_first.shape[1])
        else torch.zeros((batch,), dtype=controls.dtype, device=controls.device)
    )
    root_start_anchor = (root_pos[:, 0] - nominal_root[:, 0]).square().sum(dim=-1) * 120.0
    nominal = (
        (controls - nominal_controls).square().mean(dim=(1, 2, 3)) * 0.2
        + (root_pos - nominal_root).square().mean(dim=(1, 2)) * 0.05
        + root_start_anchor
    )

    total = (
        touchdown_sem * 300.0
        + touchdown_quality * 120.0
        + swing_loss * 120.0
        + crossing_loss * 160.0
        + arc_height_loss
        + arc_lane_loss
        + arc_along_loss
        + root_progress_loss
        + root_crossing_lift
        + endpoint_progress_loss
        + sample_crossing_loss
        + reach_loss
        + body_loss
        + root_loss
        + root_z_jump_loss * 180.0
        + root_risk_loss * 80.0
        + attitude_loss
        + smooth * 40.0
        + foot_jump_loss * 260.0
        + joint_smooth * 18.0
        + joint_jump_loss * 95.0
        + nominal
    )
    diagnostics = {
        "qp_coupled_loss_total": total.detach(),
        "qp_coupled_touchdown_semantic_loss": touchdown_sem.detach(),
        "qp_coupled_touchdown_quality_loss": touchdown_quality.detach(),
        "qp_coupled_swing_loss": swing_loss.detach(),
        "qp_coupled_terrain_clearance_loss": terrain_clearance_loss.detach(),
        "qp_coupled_crossing_loss": crossing_loss.detach(),
        "qp_coupled_low_small_arc_height_loss": arc_height_loss.detach(),
        "qp_coupled_low_small_arc_lane_loss": arc_lane_loss.detach(),
        "qp_coupled_low_small_arc_along_loss": arc_along_loss.detach(),
        "qp_coupled_low_small_progress_loss": root_progress_loss.detach(),
        "qp_coupled_low_small_root_lift_loss": root_crossing_lift.detach(),
        "qp_coupled_low_small_endpoint_progress_loss": endpoint_progress_loss.detach(),
        "qp_coupled_low_small_sample_crossing_loss": sample_crossing_loss.detach(),
        "qp_coupled_readback_loss": readback.square().mean(dim=(1, 2)).detach(),
        "qp_coupled_body_loss": body_loss.detach(),
        "qp_coupled_root_loss": root_loss.detach(),
        "qp_coupled_root_z_jump_loss": root_z_jump_loss.detach(),
        "qp_coupled_continuity_loss": (smooth * 40.0 + foot_jump_loss * 260.0 + joint_smooth * 18.0 + joint_jump_loss * 95.0).detach(),
        "qp_continuous_root_terrain_risk_reduces_progress": root_risky.detach(),
        "qp_continuous_root_progress_scale_min": root_progress_scale.detach(),
        "qp_continuous_low_small_progress_update_count": (progress_deficit.detach() > 1.0e-6).to(dtype=controls.dtype),
        "qp_continuous_low_small_progress_deficit_before_max": progress_deficit.detach().to(dtype=controls.dtype),
        "qp_continuous_low_small_foot_over_update_count": torch.zeros((batch,), dtype=controls.dtype, device=controls.device),
        "qp_coupled_readback_error_before_max": _row_max(readback.detach(), batch),
        "qp_coupled_joint_violation_before_max": _row_max(joint_violation.detach(), batch),
        "qp_coupled_reachability_violation_before_max": _row_max(reach_violation.detach(), batch),
        "qp_coupled_crossing_leg_count": _row_count(crossing_active.detach(), batch, dtype=controls.dtype),
        "qp_continuous_low_small_crossing_leg_count": _row_count(crossing_active.detach(), batch, dtype=controls.dtype),
        "qp_continuous_solver_fk_readback_error_before_max": _row_max(readback.detach(), batch),
        "qp_continuous_solver_fk_endpoint_error_before_max": _row_max(readback[:, -1].detach(), batch),
        "qp_continuous_solver_joint_limit_violation_before_max": _row_max(joint_violation.detach(), batch),
        "qp_continuous_solver_swing_clearance_deficit_before_max": _row_max(low_small_clearance.detach(), batch),
    }
    return total.sum(), diagnostics


def coupled_qp_update(
    controls: ContinuousTrajectoryControls,
    terrain: MpcPlannerTerrain,
    cfg: MpcQpPlannerCfg,
    *,
    command: Tensor | None,
    contact_state: Tensor | None,
) -> tuple[ContinuousTrajectoryControls, dict[str, Tensor]]:
    base_controls = _terrain_bound_controls(torch.as_tensor(controls.foot_control_w).detach(), terrain)
    base_root = torch.as_tensor(controls.root_pos_w).detach()
    base_rpy = torch.as_tensor(controls.root_rpy).detach()
    foot_var = base_controls.clone().requires_grad_(True)
    root_var = base_root.clone().requires_grad_(True)
    rpy_var = base_rpy.clone().requires_grad_(True)
    loss, diagnostics = _loss_terms(
        foot_var,
        root_var,
        rpy_var,
        base_controls,
        base_root,
        terrain,
        command,
        cfg,
        contact_state,
    )
    grads = torch.autograd.grad(loss, (foot_var, root_var, rpy_var), allow_unused=False)
    foot_grad, root_grad, rpy_grad = grads
    foot_step = float(getattr(cfg.runtime, "continuous_fk_readback_max_step_m", 0.12))
    root_step = float(getattr(cfg.runtime, "continuous_fk_root_z_max_step_m", 0.16))
    rpy_step = 0.08

    def _bounded_point_step(grad: Tensor, max_step: float) -> Tensor:
        norm = torch.linalg.vector_norm(grad, dim=-1, keepdim=True).clamp_min(1.0e-6)
        raw = -grad / norm * min(float(max_step), 1.0)
        return raw.clamp(min=-float(max_step), max=float(max_step))

    foot_grad_scaled = foot_grad.clone()
    foot_grad_scaled[:, :, 1:3, 2] = foot_grad_scaled[:, :, 1:3, 2] * 3.0
    foot_delta = _bounded_point_step(foot_grad_scaled, foot_step)
    root_delta = _bounded_point_step(root_grad, root_step)
    rpy_delta = _bounded_point_step(rpy_grad, rpy_step)
    foot_xy_step = min(
        max(float(foot_step), float(getattr(cfg.runtime, "continuous_low_small_crossing_arc_lateral_step_m", 0.08))),
        0.06,
    )
    foot_z_step = max(float(foot_step), float(getattr(cfg.runtime, "low_small_swing_clearance_m", 0.06)) + 0.08)
    foot_delta = foot_delta.clone()
    foot_delta[:, :, :, :2] = foot_delta[:, :, :, :2].clamp(min=-foot_xy_step, max=foot_xy_step)
    foot_delta[:, :, 1:3, 2] = foot_delta[:, :, 1:3, 2].clamp(min=-foot_z_step, max=foot_z_step)
    foot_delta[:, :, (0, 3), 2] = foot_delta[:, :, (0, 3), 2].clamp(min=-foot_step, max=foot_step)
    foot_delta[:, :, 0, :] = 0.0
    updated_foot = base_controls + foot_delta
    updated_root = base_root + root_delta
    updated_rpy = base_rpy + rpy_delta
    if contact_state is not None:
        contact = torch.as_tensor(contact_state, dtype=torch.bool, device=updated_foot.device)
        stance_leg = contact.any(dim=1)
        updated_foot[:, :, 0, :] = torch.where(stance_leg.unsqueeze(-1), base_controls[:, :, 0, :], updated_foot[:, :, 0, :])
    updated_foot = _terrain_bound_controls(updated_foot, terrain)
    root_ground = height_at(terrain, updated_root[..., :2].reshape(updated_root.shape[0], -1, 2)).reshape(
        updated_root.shape[0],
        updated_root.shape[1],
    ).to(dtype=updated_root.dtype, device=updated_root.device)
    updated_root[..., 2] = torch.maximum(updated_root[..., 2], root_ground + float(getattr(cfg.runtime, "continuous_plane_root_height_min_m", 0.30)))
    updated_rpy[..., :2] = updated_rpy[..., :2].clamp(min=-0.25, max=0.25)
    foot_delta_norm = torch.linalg.vector_norm(updated_foot - base_controls, dim=-1)
    root_z_delta = torch.clamp(base_root[..., 2] - updated_root[..., 2], min=0.0)
    root_xy_delta = torch.linalg.vector_norm(updated_root[..., :2] - base_root[..., :2], dim=-1)
    endpoint_delta = torch.linalg.vector_norm(updated_foot[:, :, 3, :2] - base_controls[:, :, 3, :2], dim=-1)
    p1p2_delta = torch.maximum(foot_delta_norm[:, :, 1], foot_delta_norm[:, :, 2])
    joint_update = torch.as_tensor(diagnostics["qp_coupled_joint_violation_before_max"], dtype=base_controls.dtype, device=base_controls.device) > 1.0e-5
    diagnostics.update(
        {
            "qp_coupled_solver_active": torch.ones((base_controls.shape[0],), dtype=base_controls.dtype, device=base_controls.device),
            "qp_coupled_foot_delta_max": _row_max(torch.linalg.vector_norm(foot_delta.detach(), dim=-1), base_controls.shape[0]),
            "qp_coupled_root_delta_max": _row_max(torch.linalg.vector_norm(root_delta.detach(), dim=-1), base_controls.shape[0]),
            "qp_coupled_rpy_delta_max": _row_max(torch.linalg.vector_norm(rpy_delta.detach(), dim=-1), base_controls.shape[0]),
            "qp_continuous_solver_update_count": _row_count(
                torch.logical_or(
                    torch.as_tensor(diagnostics["qp_coupled_touchdown_semantic_loss"], device=base_controls.device)[:, None] > 1.0e-8,
                    endpoint_delta > 1.0e-6,
                ),
                base_controls.shape[0],
                dtype=base_controls.dtype,
            ),
            "qp_continuous_solver_semantic_score_before_max": torch.as_tensor(
                diagnostics["qp_coupled_touchdown_semantic_loss"],
                dtype=base_controls.dtype,
                device=base_controls.device,
            ),
            "qp_continuous_solver_semantic_score_after_max": torch.zeros((base_controls.shape[0],), dtype=base_controls.dtype, device=base_controls.device),
            "qp_continuous_solver_foothold_score_before_max": torch.as_tensor(
                diagnostics["qp_coupled_touchdown_quality_loss"],
                dtype=base_controls.dtype,
                device=base_controls.device,
            ),
            "qp_continuous_solver_foothold_score_after_max": torch.zeros((base_controls.shape[0],), dtype=base_controls.dtype, device=base_controls.device),
            "qp_continuous_solver_swing_clearance_lift_count": _row_count(
                (updated_foot[:, :, 1, 2] - base_controls[:, :, 1, 2] > 1.0e-6)
                | (updated_foot[:, :, 2, 2] - base_controls[:, :, 2, 2] > 1.0e-6),
                base_controls.shape[0],
                dtype=base_controls.dtype,
            ),
            "qp_continuous_solver_terrain_clearance_lift_count": _row_count(p1p2_delta > 1.0e-6, base_controls.shape[0], dtype=base_controls.dtype),
            "qp_continuous_solver_fk_readback_update_count": _row_count(foot_delta_norm > 1.0e-6, base_controls.shape[0], dtype=base_controls.dtype),
            "qp_continuous_solver_fk_endpoint_update_count": _row_count(endpoint_delta > 1.0e-6, base_controls.shape[0], dtype=base_controls.dtype),
            "qp_continuous_solver_fk_root_z_update_count": _row_count(root_z_delta > 1.0e-6, base_controls.shape[0], dtype=base_controls.dtype),
            "qp_continuous_solver_fk_root_z_delta_max": _row_max(root_z_delta, base_controls.shape[0]),
            "qp_continuous_solver_fk_root_xy_update_count": _row_count(root_xy_delta > 1.0e-6, base_controls.shape[0], dtype=base_controls.dtype),
            "qp_continuous_solver_fk_root_xy_delta_max": _row_max(root_xy_delta, base_controls.shape[0]),
            "qp_continuous_solver_reachability_update_count": _row_count(foot_delta_norm > 1.0e-6, base_controls.shape[0], dtype=base_controls.dtype),
            "qp_continuous_solver_joint_limit_readback_update_count": joint_update.to(dtype=base_controls.dtype),
            "qp_continuous_solver_body_leg_clearance_update_count": _row_count(
                torch.linalg.vector_norm(root_delta.detach(), dim=-1) > 1.0e-6,
                base_controls.shape[0],
                dtype=base_controls.dtype,
            ),
            "qp_continuous_solver_body_leg_clearance_deficit_before_max": torch.as_tensor(
                diagnostics["qp_coupled_body_loss"],
                dtype=base_controls.dtype,
                device=base_controls.device,
            ),
        }
    )
    return ContinuousTrajectoryControls(
        foot_control_w=updated_foot.detach(),
        root_pos_w=updated_root.detach(),
        root_rpy=updated_rpy.detach(),
    ), diagnostics


__all__ = ["coupled_qp_update"]
