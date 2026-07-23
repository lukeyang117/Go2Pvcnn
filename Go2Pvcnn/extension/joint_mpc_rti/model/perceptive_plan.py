"""Per-leg touchdown event preview and fixed 5x5 candidate selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.model.analytic_ik import (
    go2_analytic_ik,
    go2_analytic_ik_selected,
)
from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.model.go2_kinematics import (
    HIP_OFFSETS,
    LEG_SIDE_SIGNS,
    go2_collision_geometry,
    go2_fk,
    go2_selected_leg_collision_geometry,
)
from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import (
    query_inflated_height_world,
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
    ranked_index: Tensor
    target_w: Tensor
    event_step: Tensor
    preview_touchdown_step: Tensor
    selected_sweep_safe: Tensor
    valid: Tensor
    latched: Tensor
    small_cross_required: Tensor
    small_after_mask: Tensor
    candidate_region: JointMpcTouchdownRegion
    region: JointMpcTouchdownRegion
    preview_candidate_w: Tensor
    preview_safe_mask: Tensor
    preview_score: Tensor
    preview_selected_index: Tensor
    preview_ranked_index: Tensor
    preview_target_w: Tensor
    preview_selected_sweep_safe: Tensor
    preview_valid: Tensor
    preview_candidate_region: JointMpcTouchdownRegion
    preview_region: JointMpcTouchdownRegion
    score_components: dict[str, Tensor]
    valid_components: dict[str, Tensor]
    preview_valid_components: dict[str, Tensor]

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
        (corner_xy, (corner_height + float(cfg.gait.foot_contact_offset))[..., None]), dim=-1
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
    start_root: Tensor,
    start_rpy: Tensor,
    start_joint: Tensor,
    root_target: Tensor,
    rpy_target: Tensor,
    candidate_joint: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    samples_override: int | None = None,
) -> tuple[Tensor, Tensor]:
    batch, legs, candidates = map(int, candidate_joint.shape[:3])
    current_leg = start_joint.reshape(batch, legs, 1, 1, 4, 3)
    leg_selector = torch.eye(4, dtype=torch.bool, device=start_root.device).view(
        1, 4, 1, 1, 4, 1
    )
    endpoint_selector = leg_selector.squeeze(3)
    end_leg = torch.where(
        endpoint_selector,
        candidate_joint.unsqueeze(3),
        current_leg.squeeze(3),
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
    start_state = torch.cat((start_root, start_rpy, start_joint), dim=-1)
    start = start_state[:, :, None].expand(-1, -1, candidates, -1)
    samples = (
        int(cfg.touchdown.swing_samples)
        if samples_override is None
        else int(samples_override)
    )
    fraction = constant_like(
        start,
        f"selector_swing_fraction_{samples}",
        tuple(index / float(samples - 1) for index in range(samples)),
    ).view(1, 1, 1, samples, 1)
    root_start = start_root[:, :, None, None]
    root = (
        root_start + fraction * (root_target[:, :, None, None] - root_start)
    ).expand(-1, -1, candidates, -1, -1)
    rpy_start = start_rpy[:, :, None, None]
    rpy_delta = rpy_target[:, :, None, None] - rpy_start
    yaw_delta = torch.remainder(rpy_delta[..., 2:3] + torch.pi, 2.0 * torch.pi) - torch.pi
    rpy_delta = torch.cat((rpy_delta[..., :2], yaw_delta), dim=-1)
    rpy = (rpy_start + fraction * rpy_delta).expand(-1, -1, candidates, -1, -1)

    start_geometry = go2_collision_geometry(
        start_root, start_rpy, start_joint
    )
    lift_index = torch.arange(legs, device=start.device).view(1, legs, 1, 1)
    lift = torch.gather(
        start_geometry.foot_center_w,
        2,
        lift_index.expand(batch, legs, 1, 3),
    ).squeeze(2)[:, :, None, None]
    endpoint_geometry = go2_collision_geometry(
        end_state[..., :3], end_state[..., 3:6], end_state[..., 6:]
    )
    endpoint_index = torch.arange(legs, device=start_root.device).view(1, legs, 1, 1, 1)
    endpoint_index = endpoint_index.expand(batch, legs, candidates, 1, 3)
    target = torch.gather(
        endpoint_geometry.foot_center_w, 3, endpoint_index
    ).squeeze(3)[:, :, :, None]

    tau = fraction
    smooth = 10.0 * tau.pow(3) - 15.0 * tau.pow(4) + 6.0 * tau.pow(5)
    bump = 64.0 * tau.pow(3) * (1.0 - tau).pow(3)
    side = constant_like(start, "selector_leg_side", LEG_SIDE_SIGNS).view(1, legs, 1, 1)
    endpoint_yaw = rpy_target[..., 2][:, :, None, None]
    outward = torch.stack(
        (-torch.sin(endpoint_yaw) * side, torch.cos(endpoint_yaw) * side), dim=-1
    )
    foot_xy = (
        lift[..., :2]
        + smooth * (target[..., :2] - lift[..., :2])
        + bump * float(cfg.nominal.swing_outward_offset_m) * outward
    )
    apex_query = query_perceptive_world(
        field, foot_xy.reshape(batch, legs * candidates * samples, 2)
    )
    apex_height = apex_query.inflated_height_w[..., 0].reshape(
        batch, legs, candidates, samples
    )
    apex_valid = apex_query.valid.reshape(batch, legs, candidates, samples)
    apex_height = torch.where(
        apex_valid, apex_height, torch.full_like(apex_height, -torch.inf)
    )
    apex_z = (
        apex_height.amax(dim=-1)
        + float(cfg.terrain.foot_radius_m)
        + float(cfg.nominal.swing_apex_margin_m)
    )[..., None, None]
    apex_z = torch.maximum(
        apex_z,
        torch.maximum(lift[..., 2:3], target[..., 2:3]),
    )
    first_tau = (2.0 * tau).clamp(0.0, 1.0)
    second_tau = (2.0 * tau - 1.0).clamp(0.0, 1.0)
    first_smooth = (
        10.0 * first_tau.pow(3)
        - 15.0 * first_tau.pow(4)
        + 6.0 * first_tau.pow(5)
    )
    second_smooth = (
        10.0 * second_tau.pow(3)
        - 15.0 * second_tau.pow(4)
        + 6.0 * second_tau.pow(5)
    )
    foot_z = torch.where(
        tau <= 0.5,
        lift[..., 2:3] + first_smooth * (apex_z - lift[..., 2:3]),
        apex_z + second_smooth * (target[..., 2:3] - apex_z),
    )
    foot_path = torch.cat((foot_xy, foot_z), dim=-1)

    leg_index = torch.arange(legs, device=start_root.device).view(1, legs, 1, 1)
    leg_index = leg_index.expand(batch, legs, candidates, samples)
    selected_joint, selected_reachable = go2_analytic_ik_selected(
        root, rpy, foot_path, leg_index
    )
    geometry = go2_selected_leg_collision_geometry(
        root, rpy, selected_joint, leg_index
    )
    foot = geometry.foot_center_w
    knee = geometry.knee_center_w
    calf_endpoints = geometry.calf_endpoints_w
    thigh_endpoints = geometry.thigh_endpoints_w
    capsule_samples = int(cfg.touchdown.selector_capsule_samples)
    capsule_fraction = constant_like(
        root,
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
        inflated_height, valid_query = query_inflated_height_world(
            field, flattened, channel=channel
        )
        clearance = flattened[..., 2] - inflated_height - float(vertical)
        safe = valid_query & (clearance >= 0.0)
        return safe.reshape(batch, legs, candidates, points_per_candidate).all(dim=-1)

    def segment_resolved(points: Tensor) -> Tensor:
        motion = torch.linalg.vector_norm(
            points[:, :, :, 1:] - points[:, :, :, :-1], dim=-1
        )
        return (
            motion.reshape(batch, legs, candidates, -1).amax(dim=-1)
            <= float(field.resolution)
        )

    terrain = cfg.terrain
    lower = constant_like(root, "selector_path_joint_lower", JOINT_LOWER)
    upper = constant_like(root, "selector_path_joint_upper", JOINT_UPPER)
    joint_safe = (
        (selected_joint >= lower + float(cfg.touchdown.joint_margin_rad))
        & (selected_joint <= upper - float(cfg.touchdown.joint_margin_rad))
    ).all(dim=(-1, -2))
    resolved = (
        segment_resolved(foot)
        & segment_resolved(knee)
        & segment_resolved(calf)
        & segment_resolved(thigh)
    )
    safe = (
        apex_valid.all(dim=-1)
        & selected_reachable.all(dim=-1)
        & joint_safe
        & resolved
        & part_safe(foot, 0, terrain.foot_radius_m)
        & part_safe(knee, 1, terrain.knee_radius_m + terrain.link_margin_m)
        & part_safe(calf, 2, terrain.calf_radius_m + terrain.link_margin_m)
        & part_safe(thigh, 3, terrain.thigh_radius_m + terrain.link_margin_m)
    )
    return safe, resolved


def _candidate_support_joint_safe(
    warm: Tensor,
    candidate_w: Tensor,
    event_step: Tensor,
    cfg: JointMpcRtiCfg,
) -> Tensor:
    batch, legs, candidates = map(int, candidate_w.shape[:3])
    offsets = constant_like(
        warm,
        "selector_support_node_offsets",
        tuple(range(int(cfg.gait.stance_steps) + 1)),
    ).to(torch.long)
    node = (
        event_step[:, None]
        + offsets.view(1, -1, 1)
    ).clamp_max(int(cfg.runtime.horizon_steps))
    samples = int(offsets.numel())

    def gather_root(values: Tensor) -> Tensor:
        expanded = values.unsqueeze(2).expand(-1, -1, legs, -1)
        index = node[..., None].expand(-1, -1, -1, int(values.shape[-1]))
        return torch.gather(expanded, 1, index).permute(0, 2, 1, 3)

    root = gather_root(warm[..., :3])[:, :, None].expand(
        -1, -1, candidates, -1, -1
    )
    rpy = gather_root(warm[..., 3:6])[:, :, None].expand_as(root)
    target = candidate_w[..., None, :].expand(-1, -1, -1, samples, -1)
    leg_index = torch.arange(legs, device=warm.device).view(1, legs, 1, 1)
    leg_index = leg_index.expand(batch, legs, candidates, samples)
    joint, reachable = go2_analytic_ik_selected(root, rpy, target, leg_index)
    lower = constant_like(warm, "selector_support_joint_lower", JOINT_LOWER)
    upper = constant_like(warm, "selector_support_joint_upper", JOINT_UPPER)
    margin = float(cfg.touchdown.joint_margin_rad)
    joint_safe = ((joint >= lower + margin) & (joint <= upper - margin)).all(
        dim=(-1, -2)
    )
    return reachable.all(dim=-1) & joint_safe


def select_touchdowns(
    measured: JointMpcRtiState,
    command_body: Tensor,
    schedule: FixedTrotSchedule,
    warm_nodes: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    previous_plan: TouchdownPlan | None = None,
    stage_profiler=None,
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
    candidate_z = query.height_w.reshape(batch, 4, 25) + float(cfg.gait.foot_contact_offset)
    candidate_w = torch.cat((candidate_xy, candidate_z[..., None]), dim=-1)
    if stage_profiler is not None:
        stage_profiler.begin_region()
    candidate_regions = _build_candidate_regions(
        candidate_w,
        yaw,
        field,
        cfg,
    )
    if stage_profiler is not None:
        stage_profiler.end_region()
    candidate_w = torch.cat(
        (
            candidate_xy,
            (candidate_regions.plane[..., :1] + float(cfg.gait.foot_contact_offset)),
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
    support_joint_safe = _candidate_support_joint_safe(
        warm, candidate_w, event_step, cfg
    )

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
    sweep_safe, sweep_resolved = _candidate_leg_sweep_safe(
        measured.root_pos_w[:, None].expand(-1, 4, -1),
        measured.root_rpy_w[:, None].expand(-1, 4, -1),
        measured.joint_pos[:, None].expand(-1, 4, -1),
        root_event,
        rpy_event,
        candidate_joint,
        field,
        cfg,
    )

    warm_geometry = go2_collision_geometry(warm[..., :3], warm[..., 3:6], warm[..., 6:])
    warm_target = _gather_nodes(warm_geometry.foot_center_w, event_step)
    available_lead_steps = event_step.clamp_max(int(cfg.gait.stance_steps)).to(
        warm.dtype
    )
    command_target = warm_target[..., :2] + command_axis * (
        command_speed
        * (0.5 * available_lead_steps * float(cfg.runtime.dt))
    )[..., None]
    previous_target = (
        command_target if previous_plan is None else previous_plan.target_w[..., :2]
    )
    command_score = (candidate_xy - command_target[:, :, None]).square().sum(dim=-1)
    warm_score = (candidate_xy - previous_target[:, :, None]).square().sum(dim=-1)
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
        & support_joint_safe
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

    candidate_index = constant_like(
        candidate_w,
        "touchdown_rank_candidate_index",
        tuple(float(value) for value in range(25)),
    ).to(torch.long).view(1, 1, 25)
    ranking_score = torch.where(
        candidate_index == selected_index[..., None],
        torch.full_like(masked_score, -torch.inf),
        masked_score,
    )
    ranked_index = torch.argsort(ranking_score, dim=-1, stable=True)

    selected_candidate = _gather_candidate(candidate_w, selected_index)
    selected_sweep = _gather_candidate(sweep_safe, selected_index)
    selected_region = JointMpcTouchdownRegion(
        A=_gather_candidate(candidate_regions.A, selected_index),
        b=_gather_candidate(candidate_regions.b, selected_index),
        half_extent=_gather_candidate(candidate_regions.half_extent, selected_index),
        corners_w=_gather_candidate(candidate_regions.corners_w, selected_index),
        plane=_gather_candidate(candidate_regions.plane, selected_index),
        normal_w=_gather_candidate(candidate_regions.normal_w, selected_index),
        plane_residual=_gather_candidate(candidate_regions.plane_residual, selected_index),
        area=_gather_candidate(candidate_regions.area, selected_index),
        distance_to_forbidden=_gather_candidate(
            candidate_regions.distance_to_forbidden, selected_index
        ),
        valid=_gather_candidate(candidate_regions.valid, selected_index) & valid,
    )
    selected = torch.cat(
        (
            selected_candidate[..., :2],
            (selected_region.plane[..., :1] + float(cfg.gait.foot_contact_offset)),
        ),
        dim=-1,
    )

    selected_joint = _gather_candidate(candidate_joint, selected_index)
    start_leg = measured.joint_pos.reshape(batch, 1, 4, 3).expand(-1, 4, -1, -1)
    leg_eye = constant_like(
        warm,
        "preview_start_leg_eye",
        tuple(
            tuple(float(row == column) for column in range(4))
            for row in range(4)
        ),
    ).to(torch.bool).view(1, 4, 4, 1)
    preview_start_joint = torch.where(
        leg_eye, selected_joint[:, :, None], start_leg
    ).reshape(batch, 4, 12)

    preview_node = preview_step.clamp_max(int(cfg.runtime.horizon_steps))
    preview_root = _gather_nodes(
        warm[..., :3].unsqueeze(2).expand(-1, -1, 4, -1), preview_node
    )
    preview_rpy = _gather_nodes(
        warm[..., 3:6].unsqueeze(2).expand(-1, -1, 4, -1), preview_node
    )
    preview_displacement = preview_root[..., :2] - root_event[..., :2]
    preview_yaw = preview_rpy[..., 2]
    preview_cosine = torch.cos(preview_yaw)
    preview_sine = torch.sin(preview_yaw)
    preview_hip_xy = preview_root[..., :2] + torch.stack(
        (
            preview_cosine * hip[..., 0] - preview_sine * hip[..., 1],
            preview_sine * hip[..., 0] + preview_cosine * hip[..., 1],
        ),
        dim=-1,
    )
    preview_rotated_offset = torch.stack(
        (
            preview_cosine[..., None] * offset[None, None, :, 0]
            - preview_sine[..., None] * offset[None, None, :, 1],
            preview_sine[..., None] * offset[None, None, :, 0]
            + preview_cosine[..., None] * offset[None, None, :, 1],
        ),
        dim=-1,
    )
    preview_candidate_xy = preview_hip_xy[:, :, None] + preview_rotated_offset
    preview_query = query_perceptive_world(
        field, preview_candidate_xy.reshape(batch, 100, 2)
    )
    preview_candidate_z = preview_query.height_w.reshape(batch, 4, 25)
    preview_seed = torch.cat(
        (
            preview_candidate_xy,
            (preview_candidate_z + float(cfg.gait.foot_contact_offset))[..., None],
        ),
        dim=-1,
    )
    if stage_profiler is not None:
        stage_profiler.begin_region()
    preview_regions = _build_candidate_regions(
        preview_seed, preview_yaw, field, cfg
    )
    if stage_profiler is not None:
        stage_profiler.end_region()
    preview_candidate_w = torch.cat(
        (
            preview_candidate_xy,
            (
                preview_regions.plane[..., :1]
                + float(cfg.gait.foot_contact_offset)
            ),
        ),
        dim=-1,
    )
    preview_map_safe = (
        preview_query.valid & preview_query.landing_safe
    ).reshape(batch, 4, 25)
    preview_plane_safe = (
        (preview_query.slope_rad <= float(cfg.terrain.slope_max_rad))
        & (preview_query.roughness <= float(cfg.terrain.roughness_max_m))
    ).reshape(batch, 4, 25)
    preview_root_ik = preview_root[:, :, None].expand(-1, -1, 25, -1)
    preview_rpy_ik = preview_rpy[:, :, None].expand(-1, -1, 25, -1)
    preview_targets = preview_candidate_w.unsqueeze(3).expand(-1, -1, -1, 4, -1)
    preview_all_joint, preview_all_reachable = go2_analytic_ik(
        preview_root_ik, preview_rpy_ik, preview_targets
    )
    preview_candidate_joint = (
        preview_all_joint.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)
    )
    preview_reachable = (
        preview_all_reachable.diagonal(dim1=1, dim2=3).permute(0, 2, 1)
    )
    preview_joint_safe = (
        (
            preview_candidate_joint
            >= lower + float(cfg.touchdown.joint_margin_rad)
        )
        & (
            preview_candidate_joint
            <= upper - float(cfg.touchdown.joint_margin_rad)
        )
    ).all(dim=-1)

    preview_corridor_xy = selected[:, :, None, None, :2] + corridor_fraction * (
        preview_candidate_xy[:, :, :, None] - selected[:, :, None, None, :2]
    )
    preview_corridor_query = query_perceptive_world(
        field,
        preview_corridor_xy.reshape(batch, 4 * 25 * corridor_samples, 2),
    )
    preview_small_corridor = preview_corridor_query.small_mask.reshape(
        batch, 4, 25, corridor_samples
    )
    preview_large_corridor = preview_corridor_query.large_mask.reshape(
        batch, 4, 25, corridor_samples
    )
    preview_corridor_valid = preview_corridor_query.valid.reshape(
        batch, 4, 25, corridor_samples
    ).all(dim=-1)
    preview_axis, preview_speed = _world_command_axis(
        command[:, None].expand(-1, 4, -1), preview_yaw
    )
    preview_relative = preview_corridor_xy - selected[:, :, None, None, :2]
    preview_projected = (
        preview_relative * preview_axis[:, :, None, None]
    ).sum(dim=-1)
    preview_obstacle_out = torch.where(
        preview_small_corridor,
        preview_projected,
        torch.full_like(preview_projected, -torch.inf),
    ).amax(dim=-1)
    preview_cross_required = preview_small_corridor.any(dim=-1) & (
        preview_speed[:, :, None] > 1.0e-6
    )
    preview_progress = (
        (preview_candidate_xy - selected[:, :, None, :2])
        * preview_axis[:, :, None]
    ).sum(dim=-1)
    preview_small_after = (~preview_cross_required) | (
        preview_progress
        >= preview_obstacle_out + float(cfg.touchdown.landing_after_margin_m)
    )
    preview_corridor_safe = preview_corridor_valid & ~preview_large_corridor.any(
        dim=-1
    )
    preview_sweep_safe, preview_sweep_resolved = _candidate_leg_sweep_safe(
        root_event,
        rpy_event,
        preview_start_joint,
        preview_root,
        preview_rpy,
        preview_candidate_joint,
        field,
        cfg,
        samples_override=int(cfg.touchdown.preview_swing_samples),
    )

    preview_command_target = selected[..., :2] + preview_displacement
    previous_preview_target = (
        preview_command_target
        if previous_plan is None
        else previous_plan.preview_target_w[..., :2]
    )
    preview_command_score = (
        preview_candidate_xy - preview_command_target[:, :, None]
    ).square().sum(dim=-1)
    preview_warm_score = (
        preview_candidate_xy - previous_preview_target[:, :, None]
    ).square().sum(dim=-1)
    preview_slope_score = preview_query.slope_rad.reshape(batch, 4, 25).square()
    preview_roughness_score = preview_query.roughness.reshape(batch, 4, 25).square()
    preview_edge_score = (
        preview_query.boundary_distance_m.reshape(batch, 4, 25)
        .clamp_min(0.01)
        .reciprocal()
    )
    preview_score = (
        float(cfg.touchdown.w_command) * preview_command_score
        + float(cfg.touchdown.w_warm) * preview_warm_score
        + float(cfg.touchdown.w_slope) * preview_slope_score
        + float(cfg.touchdown.w_roughness) * preview_roughness_score
        + float(cfg.touchdown.w_edge) * preview_edge_score
    )
    preview_pre_region_safe = (
        preview_map_safe
        & preview_plane_safe
        & preview_reachable
        & preview_joint_safe
        & preview_small_after
        & preview_corridor_safe
        & preview_sweep_safe
    )
    preview_safe = preview_pre_region_safe & preview_regions.valid
    preview_masked_score = torch.where(
        preview_safe, preview_score, torch.full_like(preview_score, torch.inf)
    )
    preview_selected_index = preview_masked_score.argmin(dim=-1)
    preview_valid = preview_safe.any(dim=-1)
    preview_ranking_score = torch.where(
        candidate_index == preview_selected_index[..., None],
        torch.full_like(preview_masked_score, -torch.inf),
        preview_masked_score,
    )
    preview_ranked_index = torch.argsort(
        preview_ranking_score, dim=-1, stable=True
    )
    preview_selected_candidate = _gather_candidate(
        preview_candidate_w, preview_selected_index
    )
    preview_selected_region = JointMpcTouchdownRegion(
        A=_gather_candidate(preview_regions.A, preview_selected_index),
        b=_gather_candidate(preview_regions.b, preview_selected_index),
        half_extent=_gather_candidate(
            preview_regions.half_extent, preview_selected_index
        ),
        corners_w=_gather_candidate(preview_regions.corners_w, preview_selected_index),
        plane=_gather_candidate(preview_regions.plane, preview_selected_index),
        normal_w=_gather_candidate(preview_regions.normal_w, preview_selected_index),
        plane_residual=_gather_candidate(
            preview_regions.plane_residual, preview_selected_index
        ),
        area=_gather_candidate(preview_regions.area, preview_selected_index),
        distance_to_forbidden=_gather_candidate(
            preview_regions.distance_to_forbidden, preview_selected_index
        ),
        valid=_gather_candidate(preview_regions.valid, preview_selected_index)
        & preview_valid,
    )
    preview_selected = torch.cat(
        (
            preview_selected_candidate[..., :2],
            (
                preview_selected_region.plane[..., :1]
                + float(cfg.gait.foot_contact_offset)
            ),
        ),
        dim=-1,
    )
    preview_selected_sweep = _gather_candidate(
        preview_sweep_safe, preview_selected_index
    )

    return TouchdownPlan(
        candidate_w=candidate_w,
        safe_mask=safe,
        score=score,
        selected_index=selected_index,
        ranked_index=ranked_index,
        target_w=selected,
        event_step=event_step,
        preview_touchdown_step=preview_step,
        selected_sweep_safe=selected_sweep,
        valid=valid,
        latched=latched,
        small_cross_required=small_cross_required,
        small_after_mask=small_after,
        candidate_region=JointMpcTouchdownRegion(
            A=candidate_regions.A,
            b=candidate_regions.b,
            half_extent=candidate_regions.half_extent,
            corners_w=candidate_regions.corners_w,
            plane=candidate_regions.plane,
            normal_w=candidate_regions.normal_w,
            plane_residual=candidate_regions.plane_residual,
            area=candidate_regions.area,
            distance_to_forbidden=candidate_regions.distance_to_forbidden,
            valid=candidate_regions.valid,
        ),
        region=selected_region,
        preview_candidate_w=preview_candidate_w,
        preview_safe_mask=preview_safe,
        preview_score=preview_score,
        preview_selected_index=preview_selected_index,
        preview_ranked_index=preview_ranked_index,
        preview_target_w=preview_selected,
        preview_selected_sweep_safe=preview_selected_sweep,
        preview_valid=preview_valid,
        preview_candidate_region=JointMpcTouchdownRegion(
            A=preview_regions.A,
            b=preview_regions.b,
            half_extent=preview_regions.half_extent,
            corners_w=preview_regions.corners_w,
            plane=preview_regions.plane,
            normal_w=preview_regions.normal_w,
            plane_residual=preview_regions.plane_residual,
            area=preview_regions.area,
            distance_to_forbidden=preview_regions.distance_to_forbidden,
            valid=preview_regions.valid,
        ),
        preview_region=preview_selected_region,
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
            "support_joint": support_joint_safe,
            "small_after": small_after,
            "corridor": corridor_safe,
            "sweep": sweep_safe,
            "sweep_resolved": sweep_resolved,
            "pre_region": pre_region_safe,
            "region": candidate_regions.valid,
        },
        preview_valid_components={
            "map": preview_map_safe,
            "plane": preview_plane_safe,
            "reachable": preview_reachable,
            "joint": preview_joint_safe,
            "small_after": preview_small_after,
            "corridor": preview_corridor_safe,
            "sweep": preview_sweep_safe,
            "sweep_resolved": preview_sweep_resolved,
            "pre_region": preview_pre_region_safe,
            "region": preview_regions.valid,
        },
    )


__all__ = [
    "TouchdownPlan",
    "select_touchdowns",
    "touchdown_event_steps",
]
