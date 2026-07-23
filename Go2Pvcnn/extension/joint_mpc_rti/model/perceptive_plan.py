"""Per-leg touchdown event preview and fixed 5x5 candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.analytic_ik import go2_analytic_ik
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.model.go2_kinematics import (
    HIP_OFFSETS,
    go2_collision_geometry,
)
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import query_perceptive_world
from extension.joint_mpc_rti.types import (
    JointMpcPerceptiveField,
    JointMpcRtiState,
)


JOINT_LOWER = (-1.0472, -0.6632, -2.721)
JOINT_UPPER = (1.0472, 2.966, -0.837)


@dataclass(frozen=True)
class TouchdownPlan:
    candidate_w: Tensor
    safe_mask: Tensor
    score: Tensor
    selected_index: Tensor
    target_w: Tensor
    event_step: Tensor
    preview_touchdown_step: Tensor
    selected_sweep_safe: Tensor
    valid: Tensor
    latched: Tensor
    small_cross_required: Tensor
    small_after_mask: Tensor
    score_components: dict[str, Tensor]
    valid_components: dict[str, Tensor]


def _gather_nodes(value: Tensor, node: Tensor) -> Tensor:
    batch = int(value.shape[0])
    legs = int(node.shape[1])
    batch_index = torch.arange(batch, device=value.device)[:, None]
    leg_index = torch.arange(legs, device=value.device)[None]
    return value[batch_index, node, leg_index]


def touchdown_event_steps(schedule: FixedTrotSchedule) -> tuple[Tensor, Tensor]:
    phase0 = schedule.phase_node[:, 0]
    first = torch.remainder(12 - phase0, 24)
    first = torch.where(first == 0, first.new_full((), 24), first)
    phase_horizon = schedule.phase_node[:, -1]
    tail = torch.remainder(12 - phase_horizon, 24)
    tail = torch.where(tail == 0, tail.new_full((), 24), tail)
    preview = torch.where(phase_horizon < 12, tail + int(schedule.phase_node.shape[1] - 1), -1)
    return first, preview


def _world_command_axis(command: Tensor, yaw: Tensor) -> tuple[Tensor, Tensor]:
    speed = torch.linalg.vector_norm(command[..., :2], dim=-1)
    body_axis = command[..., :2] / speed[..., None].clamp_min(1.0e-6)
    fallback = torch.stack((torch.ones_like(speed), torch.zeros_like(speed)), dim=-1)
    body_axis = torch.where((speed > 1.0e-6)[..., None], body_axis, fallback)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    world = torch.stack(
        (
            cosine * body_axis[..., 0] - sine * body_axis[..., 1],
            sine * body_axis[..., 0] + cosine * body_axis[..., 1],
        ),
        dim=-1,
    )
    return world, speed


def _candidate_leg_sweep_safe(
    measured: JointMpcRtiState,
    root_target: Tensor,
    rpy_target: Tensor,
    candidate_joint: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    batch, legs, candidates = map(int, candidate_joint.shape[:3])
    current_leg = measured.joint_pos.reshape(batch, 1, 1, 4, 3)
    leg_selector = torch.eye(4, dtype=torch.bool, device=measured.device).view(1, 4, 1, 4, 1)
    end_leg = torch.where(
        leg_selector,
        candidate_joint.unsqueeze(3),
        current_leg,
    )
    end_joint = end_leg.reshape(batch, legs, candidates, 12)
    end_state = torch.cat(
        (
            root_target[:, :, None].expand(-1, -1, candidates, -1),
            rpy_target[:, :, None].expand(-1, -1, candidates, -1),
            end_joint,
        ),
        dim=-1,
    )
    start = measured.as_vector()[:, None, None].expand(-1, legs, candidates, -1)
    samples = int(cfg.touchdown.swing_samples)
    fraction = constant_like(
        start,
        f"selector_swing_fraction_{samples}",
        tuple(index / float(samples - 1) for index in range(samples)),
    ).view(1, 1, 1, samples, 1)
    state = start[:, :, :, None] + fraction * (end_state[:, :, :, None] - start[:, :, :, None])
    geometry = go2_collision_geometry(state[..., :3], state[..., 3:6], state[..., 6:])
    def select_leg(points: Tensor) -> Tensor:
        index_shape = (1, legs, 1, 1, 1) + (1,) * (points.ndim - 5)
        leg_index = torch.arange(legs, device=state.device).view(index_shape)
        index = leg_index.expand(
            batch, legs, candidates, samples, 1, *points.shape[5:]
        )
        return torch.gather(points, 4, index).squeeze(4)

    foot = select_leg(geometry.foot_center_w)
    knee = select_leg(geometry.knee_center_w)
    calf_endpoints = select_leg(geometry.calf_endpoints_w)
    thigh_endpoints = select_leg(geometry.thigh_endpoints_w)
    capsule_samples = int(cfg.touchdown.selector_capsule_samples)
    capsule_fraction = constant_like(
        state,
        f"selector_capsule_fraction_{capsule_samples}",
        tuple(index / float(capsule_samples - 1) for index in range(capsule_samples)),
    ).view(1, 1, 1, 1, capsule_samples, 1)
    calf = calf_endpoints[..., :1, :] + capsule_fraction * (
        calf_endpoints[..., 1:, :] - calf_endpoints[..., :1, :]
    )
    thigh = thigh_endpoints[..., :1, :] + capsule_fraction * (
        thigh_endpoints[..., 1:, :] - thigh_endpoints[..., :1, :]
    )

    def part_safe(points: Tensor, channel: int, vertical: float) -> Tensor:
        points_per_candidate = int(points.numel() // (batch * legs * candidates * 3))
        flattened = points.reshape(batch, legs * candidates * points_per_candidate, 3)
        query = query_perceptive_world(field, flattened)
        clearance = flattened[..., 2] - query.inflated_height_w[..., channel] - float(vertical)
        safe = query.valid & (clearance >= 0.0)
        return safe.reshape(batch, legs, candidates, points_per_candidate).all(dim=-1)

    terrain = cfg.terrain
    return (
        part_safe(foot, 0, terrain.foot_radius_m)
        & part_safe(knee, 1, terrain.knee_radius_m + terrain.link_margin_m)
        & part_safe(calf, 2, terrain.calf_radius_m + terrain.link_margin_m)
        & part_safe(thigh, 3, terrain.thigh_radius_m + terrain.link_margin_m)
    )


def select_touchdowns(
    measured: JointMpcRtiState,
    command_body: Tensor,
    schedule: FixedTrotSchedule,
    warm_nodes: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    previous_plan: TouchdownPlan | None = None,
) -> TouchdownPlan:
    command = torch.as_tensor(command_body, dtype=measured.root_pos_w.dtype, device=measured.device)
    warm = torch.as_tensor(warm_nodes, dtype=measured.root_pos_w.dtype, device=measured.device)
    batch = measured.batch_size
    if command.shape != (batch, 3) or warm.shape != (batch, 31, 18):
        raise ValueError("command_body and warm_nodes must have shapes [B,3] and [B,31,18]")
    event_step, preview_step = touchdown_event_steps(schedule)
    root_event = _gather_nodes(warm[..., :3].unsqueeze(2).expand(-1, -1, 4, -1), event_step)
    rpy_event = _gather_nodes(warm[..., 3:6].unsqueeze(2).expand(-1, -1, 4, -1), event_step)
    yaw = rpy_event[..., 2]
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    hip = constant_like(warm, "selector_hip_offsets", HIP_OFFSETS).view(1, 4, 3)
    hip_xy = root_event[..., :2] + torch.stack(
        (
            cosine * hip[..., 0] - sine * hip[..., 1],
            sine * hip[..., 0] + cosine * hip[..., 1],
        ),
        dim=-1,
    )

    offset = constant_like(
        warm,
        "touchdown_candidate_xy",
        tuple(
            (x_value, y_value)
            for x_value in cfg.touchdown.candidate_x_m
            for y_value in cfg.touchdown.candidate_y_m
        ),
    )
    rotated_offset = torch.stack(
        (
            cosine[..., None] * offset[None, None, :, 0]
            - sine[..., None] * offset[None, None, :, 1],
            sine[..., None] * offset[None, None, :, 0]
            + cosine[..., None] * offset[None, None, :, 1],
        ),
        dim=-1,
    )
    candidate_xy = hip_xy[:, :, None] + rotated_offset
    query = query_perceptive_world(field, candidate_xy.reshape(batch, 100, 2))
    candidate_z = query.height_w.reshape(batch, 4, 25) + float(cfg.terrain.foot_radius_m)
    candidate_w = torch.cat((candidate_xy, candidate_z[..., None]), dim=-1)
    map_safe = (query.valid & query.landing_safe).reshape(batch, 4, 25)
    plane_safe = (
        (query.slope_rad <= float(cfg.terrain.slope_max_rad))
        & (query.roughness <= float(cfg.terrain.roughness_max_m))
    ).reshape(batch, 4, 25)

    root_ik = root_event[:, :, None].expand(-1, -1, 25, -1)
    rpy_ik = rpy_event[:, :, None].expand(-1, -1, 25, -1)
    repeated_targets = candidate_w.unsqueeze(3).expand(-1, -1, -1, 4, -1)
    all_joint, all_reachable = go2_analytic_ik(root_ik, rpy_ik, repeated_targets)
    candidate_joint = all_joint.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)
    reachable = all_reachable.diagonal(dim1=1, dim2=3).permute(0, 2, 1)
    lower = constant_like(warm, "selector_joint_lower", JOINT_LOWER)
    upper = constant_like(warm, "selector_joint_upper", JOINT_UPPER)
    joint_safe = (
        (candidate_joint >= lower + float(cfg.touchdown.joint_margin_rad))
        & (candidate_joint <= upper - float(cfg.touchdown.joint_margin_rad))
    ).all(dim=-1)

    measured_geometry = go2_collision_geometry(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    )
    lift = measured_geometry.foot_center_w
    corridor_samples = int(cfg.touchdown.corridor_samples)
    corridor_fraction = constant_like(
        warm,
        f"touchdown_corridor_fraction_{corridor_samples}",
        tuple(index / float(corridor_samples - 1) for index in range(corridor_samples)),
    ).view(1, 1, 1, corridor_samples, 1)
    corridor_xy = lift[:, :, None, None, :2] + corridor_fraction * (
        candidate_xy[:, :, :, None] - lift[:, :, None, None, :2]
    )
    corridor_query = query_perceptive_world(
        field, corridor_xy.reshape(batch, 4 * 25 * corridor_samples, 2)
    )
    small_corridor = corridor_query.small_mask.reshape(batch, 4, 25, corridor_samples)
    large_corridor = corridor_query.large_mask.reshape(batch, 4, 25, corridor_samples)
    corridor_valid = corridor_query.valid.reshape(batch, 4, 25, corridor_samples).all(dim=-1)
    command_axis, command_speed = _world_command_axis(command[:, None].expand(-1, 4, -1), yaw)
    relative_corridor = corridor_xy - lift[:, :, None, None, :2]
    projected_corridor = (relative_corridor * command_axis[:, :, None, None]).sum(dim=-1)
    obstacle_out = torch.where(
        small_corridor,
        projected_corridor,
        torch.full_like(projected_corridor, -torch.inf),
    ).amax(dim=-1)
    small_cross_required = small_corridor.any(dim=-1) & (command_speed[:, :, None] > 1.0e-6)
    candidate_progress = (
        (candidate_xy - lift[:, :, None, :2]) * command_axis[:, :, None]
    ).sum(dim=-1)
    small_after = (~small_cross_required) | (
        candidate_progress >= obstacle_out + float(cfg.touchdown.landing_after_margin_m)
    )
    corridor_safe = corridor_valid & ~large_corridor.any(dim=-1)
    sweep_safe = _candidate_leg_sweep_safe(
        measured,
        root_event,
        rpy_event,
        candidate_joint,
        field,
        cfg,
    )

    warm_geometry = go2_collision_geometry(warm[..., :3], warm[..., 3:6], warm[..., 6:])
    warm_target = _gather_nodes(warm_geometry.foot_center_w, event_step)
    command_target = warm_target[..., :2] + command_axis * (
        command_speed * event_step.to(warm.dtype) * float(cfg.runtime.dt)
        * float(cfg.touchdown.command_prediction_scale)
    )[..., None]
    previous_target = warm_target if previous_plan is None else previous_plan.target_w
    command_score = (candidate_xy - command_target[:, :, None]).square().sum(dim=-1)
    warm_score = (candidate_xy - previous_target[:, :, None, :2]).square().sum(dim=-1)
    slope_score = query.slope_rad.reshape(batch, 4, 25).square()
    roughness_score = query.roughness.reshape(batch, 4, 25).square()
    edge_score = query.boundary_distance_m.reshape(batch, 4, 25).clamp_min(0.01).reciprocal()
    score = (
        float(cfg.touchdown.w_command) * command_score
        + float(cfg.touchdown.w_warm) * warm_score
        + float(cfg.touchdown.w_slope) * slope_score
        + float(cfg.touchdown.w_roughness) * roughness_score
        + float(cfg.touchdown.w_edge) * edge_score
    )
    safe = map_safe & plane_safe & reachable & joint_safe & small_after & corridor_safe & sweep_safe
    masked_score = torch.where(safe, score, torch.full_like(score, torch.inf))
    selected_index = masked_score.argmin(dim=-1)
    selected = torch.gather(candidate_w, 2, selected_index[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
    selected_sweep = torch.gather(sweep_safe, 2, selected_index[..., None])[:, :, 0]
    valid = safe.any(dim=-1)

    phase0 = schedule.phase_node[:, 0]
    latched = (phase0 < 12) & (phase0 >= int(cfg.touchdown.latch_phase))
    if previous_plan is not None:
        previous_query = query_perceptive_world(field, previous_plan.target_w.reshape(batch, 4, 3))
        previous_safe = (previous_query.valid & previous_query.landing_safe).reshape(batch, 4)
        previous_distance = (
            candidate_w - previous_plan.target_w[:, :, None]
        ).square().sum(dim=-1)
        previous_index = previous_distance.argmin(dim=-1)
        previous_matches = torch.gather(
            previous_distance, 2, previous_index[..., None]
        )[..., 0] <= 1.0e-10
        previous_current_safe = torch.gather(
            safe, 2, previous_index[..., None]
        )[..., 0]
        keep = (
            latched
            & previous_plan.valid
            & previous_safe
            & previous_matches
            & previous_current_safe
        )
        selected = torch.where(keep[..., None], previous_plan.target_w, selected)
        valid = torch.where(keep, previous_plan.valid, valid)
        selected_sweep = torch.where(keep, previous_plan.selected_sweep_safe, selected_sweep)

    return TouchdownPlan(
        candidate_w=candidate_w,
        safe_mask=safe,
        score=score,
        selected_index=selected_index,
        target_w=selected,
        event_step=event_step,
        preview_touchdown_step=preview_step,
        selected_sweep_safe=selected_sweep,
        valid=valid,
        latched=latched,
        small_cross_required=small_cross_required,
        small_after_mask=small_after,
        score_components={
            "command": command_score,
            "warm": warm_score,
            "slope": slope_score,
            "roughness": roughness_score,
            "edge": edge_score,
        },
        valid_components={
            "map": map_safe,
            "plane": plane_safe,
            "reachable": reachable,
            "joint": joint_safe,
            "small_after": small_after,
            "corridor": corridor_safe,
            "sweep": sweep_safe,
        },
    )


__all__ = [
    "TouchdownPlan",
    "select_touchdowns",
    "touchdown_event_steps",
]
