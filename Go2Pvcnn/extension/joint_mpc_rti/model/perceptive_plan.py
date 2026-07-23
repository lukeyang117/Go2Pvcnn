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
from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import (
    query_landing_region_world,
    query_perceptive_world,
)
from extension.joint_mpc_rti.types import (
    JointMpcPerceptiveField,
    JointMpcRtiState,
    JointMpcTouchdownRegion,
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
    region: JointMpcTouchdownRegion
    score_components: dict[str, Tensor]
    valid_components: dict[str, Tensor]

    @property
    def region_A(self) -> Tensor:
        return self.region.A

    @property
    def region_b(self) -> Tensor:
        return self.region.b

    @property
    def region_half_extent(self) -> Tensor:
        return self.region.half_extent

    @property
    def region_corners_w(self) -> Tensor:
        return self.region.corners_w

    @property
    def region_plane(self) -> Tensor:
        return self.region.plane

    @property
    def region_normal_w(self) -> Tensor:
        return self.region.normal_w

    @property
    def region_plane_residual(self) -> Tensor:
        return self.region.plane_residual

    @property
    def region_area(self) -> Tensor:
        return self.region.area

    @property
    def region_distance_to_forbidden(self) -> Tensor:
        return self.region.distance_to_forbidden

    @property
    def region_valid(self) -> Tensor:
        return self.region.valid


@dataclass(frozen=True)
class _CandidateRegions:
    A: Tensor
    b: Tensor
    half_extent: Tensor
    corners_w: Tensor
    plane: Tensor
    normal_w: Tensor
    plane_residual: Tensor
    area: Tensor
    distance_to_forbidden: Tensor
    valid: Tensor


def _prefix_true_count(value: Tensor) -> Tensor:
    return torch.cumprod(value.to(dtype=torch.int32), dim=-1).sum(dim=-1)


def _gather_candidate(value: Tensor, index: Tensor) -> Tensor:
    trailing = tuple(value.shape[3:])
    gather_index = index[..., None]
    for _ in trailing:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(*index.shape, 1, *trailing)
    return torch.gather(value, 2, gather_index).squeeze(2)


def _build_candidate_regions(
    candidate_w: Tensor,
    yaw: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    cap_m: float | None = None,
) -> _CandidateRegions:
    batch, legs, candidates = map(int, candidate_w.shape[:3])
    resolution = float(field.resolution)
    region_cap = float(cfg.region.cap_m) if cap_m is None else float(cap_m)
    steps = int(round(region_cap / resolution))
    if steps < 1 or abs(steps * resolution - region_cap) > 1.0e-6:
        raise ValueError("region cap must be a positive multiple of field resolution")
    side = 2 * steps + 1
    index = constant_like(
        candidate_w,
        f"region_grid_index_{steps}",
        tuple(float(value) for value in range(-steps, steps + 1)),
    )
    local_x = index.view(side, 1) * resolution
    local_y = index.view(1, side) * resolution
    cosine = torch.cos(yaw)[..., None, None]
    sine = torch.sin(yaw)[..., None, None]
    offset_x = cosine * local_x - sine * local_y
    offset_y = sine * local_x + cosine * local_y
    offset_w = torch.stack((offset_x, offset_y), dim=-1)
    sample_xy = candidate_w[..., None, None, :2] + offset_w[:, :, None]
    query = query_landing_region_world(
        field, sample_xy.reshape(batch, legs * candidates * side * side, 2)
    )
    sample_shape = (batch, legs, candidates, side, side)
    cell_safe = (
        query.valid
        & query.landing_safe
        & ~query.semantic_edge_mask
        & (query.slope_rad <= float(cfg.terrain.slope_max_rad))
        & (query.roughness <= float(cfg.terrain.roughness_max_m))
    ).reshape(sample_shape)
    center_safe = cell_safe[..., steps, steps]

    center_x_safe = cell_safe[..., steps, :]
    left_count = _prefix_true_count(
        torch.flip(cell_safe[..., :steps, :], dims=(-2,)).transpose(-2, -1)
    )
    right_count = _prefix_true_count(
        cell_safe[..., steps + 1 :, :].transpose(-2, -1)
    )
    left_count = torch.where(center_x_safe, left_count, torch.zeros_like(left_count))
    right_count = torch.where(center_x_safe, right_count, torch.zeros_like(right_count))
    center_left = left_count[..., steps : steps + 1]
    center_right = right_count[..., steps : steps + 1]
    left_negative = torch.cummin(
        torch.cat((center_left, torch.flip(left_count[..., :steps], dims=(-1,))), dim=-1),
        dim=-1,
    ).values
    left_positive = torch.cummin(
        torch.cat((center_left, left_count[..., steps + 1 :]), dim=-1), dim=-1
    ).values
    right_negative = torch.cummin(
        torch.cat((center_right, torch.flip(right_count[..., :steps], dims=(-1,))), dim=-1),
        dim=-1,
    ).values
    right_positive = torch.cummin(
        torch.cat((center_right, right_count[..., steps + 1 :]), dim=-1), dim=-1
    ).values
    y_pair = constant_like(
        candidate_w,
        f"region_y_pairs_{steps}",
        tuple(
            (float(negative), float(positive))
            for negative in range(steps + 1)
            for positive in range(steps + 1)
        ),
    ).to(torch.long)
    pair_count = int(y_pair.shape[0])
    y_negative = y_pair[:, 0]
    y_positive = y_pair[:, 1]
    x_negative = torch.minimum(
        torch.index_select(left_negative, -1, y_negative),
        torch.index_select(left_positive, -1, y_positive),
    )
    x_positive = torch.minimum(
        torch.index_select(right_negative, -1, y_negative),
        torch.index_select(right_positive, -1, y_positive),
    )
    y_negative_count = y_negative.view(1, 1, 1, pair_count).expand(
        batch, legs, candidates, -1
    )
    y_positive_count = y_positive.view(1, 1, 1, pair_count).expand(
        batch, legs, candidates, -1
    )
    count_options = torch.cat(
        (
            x_negative[..., None],
            x_positive[..., None],
            y_negative_count[..., None],
            y_positive_count[..., None],
        ),
        dim=-1,
    )
    extent_options = (
        count_options.to(candidate_w.dtype) * resolution - float(cfg.region.margin_m)
    ).clamp_min(0.0)
    option_valid = center_safe[..., None] & (
        extent_options >= float(cfg.region.min_half_extent_m)
    ).all(dim=-1)
    option_area = (
        (extent_options[..., 0] + extent_options[..., 1])
        * (extent_options[..., 2] + extent_options[..., 3])
    )
    best_option = torch.where(
        option_valid, option_area, torch.full_like(option_area, -torch.inf)
    ).argmax(dim=-1)
    counts = torch.gather(
        count_options,
        -2,
        best_option[..., None, None].expand(-1, -1, -1, 1, 4),
    ).squeeze(-2)
    raw_extent = counts.to(candidate_w.dtype) * resolution
    half_extent = (raw_extent - float(cfg.region.margin_m)).clamp_min(0.0)
    x_negative_m, x_positive_m, y_negative_m, y_positive_m = half_extent.unbind(dim=-1)
    region_valid = center_safe & (
        half_extent >= float(cfg.region.min_half_extent_m)
    ).all(dim=-1)

    local_corners = torch.stack(
        (
            torch.stack((-x_negative_m, -y_negative_m), dim=-1),
            torch.stack((-x_negative_m, y_positive_m), dim=-1),
            torch.stack((x_positive_m, y_positive_m), dim=-1),
            torch.stack((x_positive_m, -y_negative_m), dim=-1),
        ),
        dim=-2,
    )
    corner_cosine = torch.cos(yaw)[..., None, None]
    corner_sine = torch.sin(yaw)[..., None, None]
    corner_x = corner_cosine * local_corners[..., 0] - corner_sine * local_corners[..., 1]
    corner_y = corner_sine * local_corners[..., 0] + corner_cosine * local_corners[..., 1]
    corner_xy = candidate_w[..., None, :2] + torch.stack((corner_x, corner_y), dim=-1)
    corner_query = query_landing_region_world(
        field, corner_xy.reshape(batch, legs * candidates * 4, 2)
    )
    corner_safe = (
        corner_query.valid
        & corner_query.landing_safe
        & ~corner_query.semantic_edge_mask
        & (corner_query.slope_rad <= float(cfg.terrain.slope_max_rad))
        & (corner_query.roughness <= float(cfg.terrain.roughness_max_m))
    ).reshape(batch, legs, candidates, 4).all(dim=-1)

    local_grid_x = local_x.expand(side, side)
    local_grid_y = local_y.expand(side, side)
    inside = (
        (local_grid_x >= -x_negative_m[..., None, None])
        & (local_grid_x <= x_positive_m[..., None, None])
        & (local_grid_y >= -y_negative_m[..., None, None])
        & (local_grid_y <= y_positive_m[..., None, None])
    )
    weights = (inside & cell_safe).to(candidate_w.dtype)
    design = torch.stack(
        (
            torch.ones_like(offset_x),
            offset_x,
            offset_y,
        ),
        dim=-1,
    )[:, :, None].expand(-1, -1, candidates, -1, -1, -1)
    height = query.height_w.reshape(sample_shape)
    center_height = height[..., steps, steps]
    height_delta = height - center_height[..., None, None]
    dx = design[..., 1]
    dy = design[..., 2]
    zero = torch.zeros_like(center_height)
    one = torch.ones_like(center_height)
    xx = (weights * dx.square()).sum(dim=(-2, -1))
    xy = (weights * dx * dy).sum(dim=(-2, -1))
    yy = (weights * dy.square()).sum(dim=(-2, -1))
    rhs_x = (weights * dx * height_delta).sum(dim=(-2, -1))
    rhs_y = (weights * dy * height_delta).sum(dim=(-2, -1))
    normal_matrix = torch.stack(
        (
            torch.stack((one, zero, zero), dim=-1),
            torch.stack((zero, xx, xy), dim=-1),
            torch.stack((zero, xy, yy), dim=-1),
        ),
        dim=-2,
    )
    normal_rhs = torch.stack((center_height, rhs_x, rhs_y), dim=-1)
    slope_regularizer = constant_like(
        candidate_w,
        "region_plane_slope_regularizer",
        ((0.0, 0.0, 0.0), (0.0, 1.0e-6, 0.0), (0.0, 0.0, 1.0e-6)),
    )
    plane = fixed_general_solve(
        normal_matrix + slope_regularizer, normal_rhs[..., None]
    )[..., 0]
    predicted_height = torch.einsum("blcgxd,blcd->blcgx", design, plane)
    residual_grid = torch.where(inside, (height - predicted_height).abs(), 0.0)
    plane_residual = residual_grid.amax(dim=(-2, -1))
    normal_w = torch.cat((-plane[..., 1:], torch.ones_like(plane[..., :1])), dim=-1)
    normal_w = normal_w / torch.linalg.vector_norm(normal_w, dim=-1, keepdim=True).clamp_min(1.0e-9)
    region_valid = region_valid & corner_safe & (
        plane_residual <= float(cfg.region.max_plane_residual_m)
    )
    corner_height = plane[..., :1] + torch.einsum(
        "blci,blcki->blck", plane[..., 1:], corner_xy - candidate_w[..., None, :2]
    )
    corners_w = torch.cat(
        (corner_xy, (corner_height + float(cfg.terrain.foot_radius_m))[..., None]), dim=-1
    )

    hip_x = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    hip_y = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
    A = torch.stack((hip_x, -hip_x, hip_y, -hip_y), dim=-2)
    center_projection_x = (hip_x[:, :, None] * candidate_w[..., :2]).sum(dim=-1)
    center_projection_y = (hip_y[:, :, None] * candidate_w[..., :2]).sum(dim=-1)
    b = torch.stack(
        (
            x_negative_m - center_projection_x,
            x_positive_m + center_projection_x,
            y_negative_m - center_projection_y,
            y_positive_m + center_projection_y,
        ),
        dim=-1,
    )
    A = A[:, :, None].expand(-1, -1, candidates, -1, -1)
    area = (x_negative_m + x_positive_m) * (y_negative_m + y_positive_m)
    distance_to_forbidden = (
        (counts.to(candidate_w.dtype) + 1.0) * resolution
    ).amin(dim=-1).clamp_max(region_cap)
    return _CandidateRegions(
        A=A,
        b=b,
        half_extent=half_extent,
        corners_w=corners_w,
        plane=plane,
        normal_w=normal_w,
        plane_residual=plane_residual,
        area=area,
        distance_to_forbidden=distance_to_forbidden,
        valid=region_valid,
    )


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
    phase0 = schedule.phase_node[:, 0]
    latched = (phase0 < 12) & (phase0 >= int(cfg.touchdown.latch_phase))
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
    if previous_plan is not None:
        candidate_index = constant_like(
            candidate_xy,
            "touchdown_candidate_index",
            tuple(float(value) for value in range(25)),
        ).view(1, 1, 25)
        inject_previous = (
            latched[..., None]
            & previous_plan.valid[..., None]
            & (candidate_index == previous_plan.selected_index[..., None])
        )
        candidate_xy = torch.where(
            inject_previous[..., None],
            previous_plan.target_w[:, :, None, :2],
            candidate_xy,
        )
    query = query_perceptive_world(field, candidate_xy.reshape(batch, 100, 2))
    candidate_z = query.height_w.reshape(batch, 4, 25) + float(cfg.terrain.foot_radius_m)
    candidate_w = torch.cat((candidate_xy, candidate_z[..., None]), dim=-1)
    minimum_region_cap = (
        round(
            (float(cfg.region.min_half_extent_m) + float(cfg.region.margin_m))
            / float(field.resolution)
        )
        * float(field.resolution)
    )
    candidate_regions = _build_candidate_regions(
        candidate_w,
        yaw,
        field,
        cfg,
        cap_m=minimum_region_cap,
    )
    candidate_w = torch.cat(
        (
            candidate_xy,
            (candidate_regions.plane[..., :1] + float(cfg.terrain.foot_radius_m)),
        ),
        dim=-1,
    )
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
    pre_region_safe = (
        map_safe
        & plane_safe
        & reachable
        & joint_safe
        & small_after
        & corridor_safe
        & sweep_safe
    )
    safe = pre_region_safe & candidate_regions.valid
    selection_safe = safe
    previous_index = None
    previous_matches = None
    previous_current_safe = None
    if previous_plan is not None:
        previous_query = query_perceptive_world(field, previous_plan.target_w.reshape(batch, 4, 3))
        previous_safe = (previous_query.valid & previous_query.landing_safe).reshape(batch, 4)
        previous_distance = (
            candidate_w[..., :2] - previous_plan.target_w[:, :, None, :2]
        ).square().sum(dim=-1)
        previous_index = previous_distance.argmin(dim=-1)
        previous_matches = torch.gather(
            previous_distance, 2, previous_index[..., None]
        )[..., 0] <= 1.0e-10
        previous_current_safe = torch.gather(
            safe, 2, previous_index[..., None]
        )[..., 0]

    masked_score = torch.where(
        selection_safe, score, torch.full_like(score, torch.inf)
    )
    selected_index = masked_score.argmin(dim=-1)
    valid = selection_safe.any(dim=-1)

    if previous_plan is not None:
        assert previous_index is not None
        assert previous_matches is not None
        assert previous_current_safe is not None
        keep = (
            latched
            & previous_plan.valid
            & previous_safe
            & previous_matches
            & previous_current_safe
        )
        selected_index = torch.where(keep, previous_index, selected_index)

    selected_candidate = _gather_candidate(candidate_w, selected_index)
    selected_sweep = _gather_candidate(sweep_safe, selected_index)
    maximum_region = _build_candidate_regions(
        selected_candidate.unsqueeze(2), yaw, field, cfg
    )
    use_maximum = maximum_region.valid[..., 0]

    def select_region_value(maximum: Tensor, minimum: Tensor) -> Tensor:
        maximum_value = maximum[:, :, 0]
        condition = use_maximum.view(
            *use_maximum.shape,
            *((1,) * (maximum_value.ndim - use_maximum.ndim)),
        )
        return torch.where(condition, maximum_value, minimum)

    selected_region = JointMpcTouchdownRegion(
        A=select_region_value(
            maximum_region.A, _gather_candidate(candidate_regions.A, selected_index)
        ),
        b=select_region_value(
            maximum_region.b, _gather_candidate(candidate_regions.b, selected_index)
        ),
        half_extent=select_region_value(
            maximum_region.half_extent,
            _gather_candidate(candidate_regions.half_extent, selected_index),
        ),
        corners_w=select_region_value(
            maximum_region.corners_w,
            _gather_candidate(candidate_regions.corners_w, selected_index),
        ),
        plane=select_region_value(
            maximum_region.plane,
            _gather_candidate(candidate_regions.plane, selected_index),
        ),
        normal_w=select_region_value(
            maximum_region.normal_w,
            _gather_candidate(candidate_regions.normal_w, selected_index),
        ),
        plane_residual=select_region_value(
            maximum_region.plane_residual,
            _gather_candidate(candidate_regions.plane_residual, selected_index),
        ),
        area=select_region_value(
            maximum_region.area,
            _gather_candidate(candidate_regions.area, selected_index),
        ),
        distance_to_forbidden=select_region_value(
            maximum_region.distance_to_forbidden,
            _gather_candidate(candidate_regions.distance_to_forbidden, selected_index),
        ),
        valid=(use_maximum | _gather_candidate(candidate_regions.valid, selected_index))
        & valid,
    )
    selected = torch.cat(
        (
            selected_candidate[..., :2],
            (selected_region.plane[..., :1] + float(cfg.terrain.foot_radius_m)),
        ),
        dim=-1,
    )

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
        region=selected_region,
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
            "pre_region": pre_region_safe,
            "region": candidate_regions.valid,
        },
    )


__all__ = [
    "TouchdownPlan",
    "select_touchdowns",
    "touchdown_event_steps",
]
