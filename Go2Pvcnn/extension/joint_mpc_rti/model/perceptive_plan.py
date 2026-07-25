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
from extension.joint_mpc_rti.model.swing_profile import (
    crossing_root_lift_offset,
    swing_height_profile,
    swing_xy_profile,
)
from extension.joint_mpc_rti.solver.fixed_general import fixed_general_solve
from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.terrain.query import (
    query_inflated_height_world,
    query_landing_region_world,
    query_perceptive_world,
    query_world,
)
from extension.joint_mpc_rti.types import (
    JointMpcPerceptiveField,
    JointMpcRtiState,
    JointMpcTerrainField,
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
    candidate_swing_offset_w: Tensor
    selected_swing_offset_w: Tensor
    preview_small_cross_required: Tensor
    preview_candidate_swing_offset_w: Tensor
    preview_selected_swing_offset_w: Tensor
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
    batch_index = torch.arange(batch, device=value.device).view(
        batch, *([1] * (node.ndim - 1))
    )
    leg_index = torch.arange(legs, device=value.device).view(
        1, legs, *([1] * (node.ndim - 2))
    )
    batch_index = batch_index.expand_as(node)
    leg_index = leg_index.expand_as(node)
    return value[batch_index, node, leg_index]


def _preview_sweep_start(
    warm: Tensor,
    primary_target_w: Tensor,
    preview_touchdown_step: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor, Tensor]:
    liftoff_step = (
        preview_touchdown_step - int(cfg.gait.swing_steps)
    ).clamp(0, int(cfg.runtime.horizon_steps))
    root = _gather_nodes(
        warm[..., :3].unsqueeze(2).expand(-1, -1, 4, -1), liftoff_step
    )
    rpy = _gather_nodes(
        warm[..., 3:6].unsqueeze(2).expand(-1, -1, 4, -1), liftoff_step
    )
    warm_joint = _gather_nodes(
        warm[..., 6:].unsqueeze(2).expand(-1, -1, 4, -1), liftoff_step
    )
    leg_index = torch.arange(4, device=warm.device).view(1, 4).expand(
        int(warm.shape[0]), -1
    )
    anchored_joint, _ = go2_analytic_ik_selected(
        root, rpy, primary_target_w, leg_index
    )
    leg_eye = constant_like(
        warm,
        "preview_start_leg_eye",
        tuple(
            tuple(float(row == column) for column in range(4))
            for row in range(4)
        ),
    ).to(torch.bool).view(1, 4, 4, 1)
    joint = torch.where(
        leg_eye,
        anchored_joint[:, :, None],
        warm_joint.reshape(int(warm.shape[0]), 4, 4, 3),
    )
    return root, rpy, joint.reshape(int(warm.shape[0]), 4, 12)


def _preview_sweep_pose_path(
    warm: Tensor,
    preview_touchdown_step: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    batch = int(warm.shape[0])
    liftoff_step = (
        preview_touchdown_step - int(cfg.gait.swing_steps)
    ).clamp(0, int(cfg.runtime.horizon_steps))
    offset = constant_like(
        warm,
        "preview_sweep_pose_path_offset",
        tuple(float(value) for value in range(int(cfg.gait.swing_steps) + 1)),
    ).to(torch.long)
    node = (liftoff_step[..., None] + offset.view(1, 1, -1)).clamp_max(
        int(cfg.runtime.horizon_steps)
    )

    def gather(value: Tensor) -> Tensor:
        expanded = value.unsqueeze(2).expand(-1, -1, 4, -1)
        index = node.permute(0, 2, 1)[..., None].expand(
            batch, int(node.shape[-1]), 4, int(value.shape[-1])
        )
        return torch.gather(expanded, 1, index).permute(0, 2, 1, 3)

    return gather(warm[..., :3]), gather(warm[..., 3:6])


def touchdown_event_steps(schedule: FixedTrotSchedule) -> tuple[Tensor, Tensor]:
    phase0 = schedule.phase_node[:, 0]
    first = torch.remainder(12 - phase0, 24)
    first = torch.where(first == 0, first.new_full((), 24), first)
    phase_horizon = schedule.phase_node[:, -1]
    tail = torch.remainder(12 - phase_horizon, 24)
    tail = torch.where(tail == 0, tail.new_full((), 24), tail)
    preview = torch.where(phase_horizon < 12, tail + int(schedule.phase_node.shape[1] - 1), -1)
    return first, preview


def _crossing_after_mask(
    lift_xy: Tensor,
    candidate_xy: Tensor,
    command_axis: Tensor,
    corridor_xy: Tensor,
    small_corridor: Tensor,
    crossing_required: Tensor,
    continued_candidate: Tensor,
    *,
    margin_m: float,
    sdf_obstacle_out: Tensor | None = None,
) -> Tensor:
    candidate_progress = (
        (candidate_xy - lift_xy[:, :, None]) * command_axis[:, :, None]
    ).sum(dim=-1)
    corridor_progress = (
        (corridor_xy - lift_xy[:, :, None, None])
        * command_axis[:, :, None, None]
    ).sum(dim=-1)
    obstacle_out = torch.where(
        small_corridor,
        corridor_progress,
        torch.full_like(corridor_progress, -torch.inf),
    ).amax(dim=-1)
    sdf_crosses = torch.zeros_like(crossing_required)
    if sdf_obstacle_out is not None:
        obstacle_out = torch.maximum(obstacle_out, sdf_obstacle_out)
        sdf_crosses = torch.isfinite(sdf_obstacle_out)
    return (~crossing_required) | (
        (small_corridor.any(dim=-1) | sdf_crosses)
        & (candidate_progress >= obstacle_out + float(margin_m))
    )


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


def _crossing_sweep_continuation(
    phase0: Tensor,
    continued_candidate: Tensor,
    *,
    swing_steps: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    current_swing = phase0 < int(swing_steps)
    continuation_mask = continued_candidate & current_swing[..., None]
    start_tau = torch.where(
        current_swing,
        phase0.to(dtype) / float(swing_steps),
        torch.zeros_like(phase0, dtype=dtype),
    )
    return continuation_mask, start_tau


def _continuation_retarget_candidates(
    candidate_xy: Tensor,
    prior_target_xy: Tensor,
    prior_index: Tensor,
    continued_crossing: Tensor,
    *,
    current_swing: Tensor,
    event_yaw: Tensor,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor, Tensor]:
    """Inject the committed target and bounded outward warm retargets."""
    offsets_m = tuple(float(value) for value in cfg.touchdown.continuation_outward_retarget_m)
    if not offsets_m or offsets_m[0] != 0.0 or len(offsets_m) > 25:
        raise ValueError(
            "continuation_outward_retarget_m must start at zero and contain at most 25 values"
        )
    candidate_index = constant_like(
        candidate_xy,
        "continuation_retarget_candidate_index",
        tuple(float(value) for value in range(25)),
    ).to(torch.long).view(1, 1, 25)
    rank = torch.remainder(candidate_index - prior_index[..., None], 25)
    variant_slot = rank < len(offsets_m)
    exact = continued_crossing[..., None] & (rank == 0)
    injected = continued_crossing[..., None] & torch.where(
        current_swing[..., None], rank == 0, variant_slot
    )
    offsets = constant_like(
        candidate_xy,
        "continuation_outward_retarget_offsets",
        offsets_m,
    )
    magnitude = offsets[rank.clamp_max(len(offsets_m) - 1)]
    side = constant_like(
        candidate_xy, "continuation_retarget_leg_side", LEG_SIDE_SIGNS
    ).view(1, 4)
    outward = torch.stack(
        (-torch.sin(event_yaw) * side, torch.cos(event_yaw) * side), dim=-1
    )
    retargeted = prior_target_xy[:, :, None] + magnitude[..., None] * outward[:, :, None]
    return torch.where(injected[..., None], retargeted, candidate_xy), injected, exact


def _small_staging_mask(
    candidate_progress: Tensor,
    obstacle_in: Tensor,
    obstacle_out: Tensor,
    *,
    before_margin_m: float,
    after_margin_m: float,
    continued_candidate: Tensor | None = None,
) -> Tensor:
    obstacle_visible = torch.isfinite(obstacle_in) & torch.isfinite(obstacle_out)
    before = candidate_progress <= obstacle_in - float(before_margin_m)
    after = candidate_progress >= obstacle_out + float(after_margin_m)
    return (~obstacle_visible) | before | after


def _committed_crossing_selection(
    safe_cross: Tensor,
    ordinary_or_safe: Tensor,
    selected_cross_leg: Tensor,
    continued_crossing: Tensor,
    continued_candidate: Tensor,
) -> Tensor:
    base = torch.where(
        selected_cross_leg[..., None], safe_cross, ordinary_or_safe
    )
    committed_safe = safe_cross & continued_candidate
    committed_leg = continued_crossing & committed_safe.any(dim=-1)
    return torch.where(committed_leg[..., None], committed_safe, base)


def _prefer_current_swing_post_obstacle(
    selection_safe: Tensor,
    safe: Tensor,
    candidate_progress: Tensor,
    obstacle_in: Tensor,
    obstacle_out: Tensor,
    *,
    current_swing: Tensor,
    selected_cross_leg: Tensor,
    continued_crossing: Tensor,
    before_margin_m: float,
    after_margin_m: float,
) -> Tensor:
    safe_before = (
        safe
        & torch.isfinite(obstacle_in)
        & (candidate_progress <= obstacle_in - float(before_margin_m))
    )
    safe_post = (
        safe
        & torch.isfinite(obstacle_out)
        & (candidate_progress >= obstacle_out + float(after_margin_m))
    )
    prefer_post = (
        current_swing
        & ~selected_cross_leg
        & ~continued_crossing
        & (safe_before.sum(dim=-1) <= 1)
        & safe_post.any(dim=-1)
    )
    return torch.where(prefer_post[..., None], safe_post, selection_safe)


def _post_obstacle_continuation(
    *,
    current_swing: Tensor,
    candidate_progress: Tensor,
    obstacle_out: Tensor,
    margin_m: float,
) -> Tensor:
    return current_swing & torch.isfinite(obstacle_out) & (
        candidate_progress >= obstacle_out + float(margin_m)
    )


def _small_crossing_offsets(
    lift_xy: Tensor,
    candidate_xy: Tensor,
    command_axis: Tensor,
    terrain_field: JointMpcTerrainField,
    cfg: JointMpcRtiCfg,
    *,
    delayed_profile: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return one bounded SDF-guided bump control point per touchdown candidate."""
    batch, legs, candidates = map(int, candidate_xy.shape[:3])
    samples = int(cfg.touchdown.corridor_samples)
    tau = constant_like(
        candidate_xy,
        f"small_cross_offset_fraction_{samples}",
        tuple(index / float(samples - 1) for index in range(samples)),
    ).view(1, 1, 1, samples, 1)
    delay = (
        float(cfg.nominal.small_cross_lateral_start_fraction)
        if delayed_profile
        else 0.0
    )
    crossing_tau = ((tau - delay) / max(1.0 - delay, 1.0e-6)).clamp(0.0, 1.0)
    smooth = (
        10.0 * crossing_tau.pow(3)
        - 15.0 * crossing_tau.pow(4)
        + 6.0 * crossing_tau.pow(5)
    )
    bump_samples = 64.0 * crossing_tau[..., 0].pow(3) * (
        1.0 - crossing_tau[..., 0]
    ).pow(3)
    baseline = lift_xy[:, :, None, None] + smooth * (
        candidate_xy[..., None, :] - lift_xy[:, :, None, None]
    )
    query = query_world(
        terrain_field, baseline.reshape(batch, legs * candidates * samples, 2)
    )
    distance = query.small_distance_m.reshape(batch, legs, candidates, samples)
    gradient = query.small_gradient_w.reshape(
        batch, legs, candidates, samples, 2
    )
    valid = query.valid.reshape(batch, legs, candidates, samples)
    interior = ((tau[..., 0] >= 0.10) & (tau[..., 0] <= 0.90)).expand(
        batch, legs, candidates, -1
    )
    eligible_distance = torch.where(
        valid & interior, distance, torch.full_like(distance, torch.inf)
    )
    closest_index = eligible_distance.argmin(dim=-1)
    closest_distance = torch.gather(
        eligible_distance, -1, closest_index[..., None]
    ).squeeze(-1)
    closest_gradient = torch.gather(
        gradient,
        -2,
        closest_index[..., None, None].expand(-1, -1, -1, 1, 2),
    ).squeeze(-2)
    closest_point = torch.gather(
        baseline,
        -2,
        closest_index[..., None, None].expand(-1, -1, -1, 1, 2),
    ).squeeze(-2)
    axis = command_axis[:, :, None]
    lateral_gradient = closest_gradient - (
        closest_gradient * axis
    ).sum(dim=-1, keepdim=True) * axis
    lateral_norm = torch.linalg.vector_norm(lateral_gradient, dim=-1, keepdim=True)
    toward_obstacle = -lateral_gradient / lateral_norm.clamp_min(1.0e-6)
    bump = torch.gather(
        bump_samples.expand(batch, legs, candidates, -1),
        -1,
        closest_index[..., None],
    ).squeeze(-1)
    desired_shift = (
        closest_distance
        - float(cfg.touchdown.small_cross_foot_overlap_fraction)
        * float(cfg.terrain.foot_radius_m)
    ).clamp_min(0.0)
    control_magnitude = (desired_shift / bump.clamp_min(0.20)).clamp_max(
        float(cfg.nominal.small_cross_lateral_offset_cap_m)
    )
    candidate_progress = (
        (candidate_xy - lift_xy[:, :, None]) * axis
    ).sum(dim=-1)
    gradient_norm = torch.linalg.vector_norm(
        closest_gradient, dim=-1, keepdim=True
    )
    toward_boundary = -closest_gradient / gradient_norm.clamp_min(1.0e-6)
    inside_seed = closest_point + toward_boundary * (
        closest_distance.clamp_min(0.0)[..., None]
        + float(terrain_field.resolution)
    )
    scan_samples = int(cfg.touchdown.small_cross_obstacle_scan_samples)
    scan_extent = float(cfg.touchdown.small_cross_obstacle_scan_extent_m)
    scan_delta = constant_like(
        candidate_xy,
        f"small_cross_obstacle_scan_{scan_samples}_{scan_extent}",
        tuple(
            -scan_extent + 2.0 * scan_extent * index / float(scan_samples - 1)
            for index in range(scan_samples)
        ),
    ).view(1, 1, 1, scan_samples, 1)
    scan_point = inside_seed[..., None, :] + scan_delta * axis[..., None, :]
    scan_query = query_world(
        terrain_field,
        scan_point.reshape(batch, legs * candidates * scan_samples, 2),
    )
    scan_inside = (
        scan_query.valid
        & (scan_query.small_distance_m <= 0.0)
    ).reshape(batch, legs, candidates, scan_samples)
    scan_progress = (
        (scan_point - lift_xy[:, :, None, None]) * axis[..., None, :]
    ).sum(dim=-1)
    obstacle_out = torch.where(
        scan_inside, scan_progress, torch.full_like(scan_progress, -torch.inf)
    ).amax(dim=-1)
    obstacle_in = torch.where(
        scan_inside, scan_progress, torch.full_like(scan_progress, torch.inf)
    ).amin(dim=-1)
    field_cosine = torch.cos(terrain_field.yaw_w)[:, None, None]
    field_sine = torch.sin(terrain_field.yaw_w)[:, None, None]
    axis_local_x = field_cosine * axis[..., 0] + field_sine * axis[..., 1]
    axis_local_y = -field_sine * axis[..., 0] + field_cosine * axis[..., 1]
    half_cell_projection = 0.5 * float(terrain_field.resolution) * (
        axis_local_x.abs() + axis_local_y.abs()
    )
    half_scan_step = scan_extent / float(scan_samples - 1)
    boundary_padding = (
        half_cell_projection
        + half_scan_step
        + 4.0 * torch.finfo(candidate_xy.dtype).eps
    )
    obstacle_in = torch.where(
        torch.isfinite(obstacle_in), obstacle_in - boundary_padding, obstacle_in
    )
    obstacle_out = torch.where(
        torch.isfinite(obstacle_out), obstacle_out + boundary_padding, obstacle_out
    )
    opportunity = (
        torch.isfinite(closest_distance)
        & torch.isfinite(obstacle_out)
        & (lateral_norm[..., 0] > 1.0e-6)
        & (candidate_progress > 0.0)
        & (
            desired_shift
            <= bump * float(cfg.nominal.small_cross_lateral_offset_cap_m)
        )
    )
    offset = toward_obstacle * control_magnitude[..., None]
    return (
        torch.where(opportunity[..., None], offset, torch.zeros_like(offset)),
        opportunity,
        obstacle_in,
        obstacle_out,
    )


def _candidate_leg_sweep_safe(
    start_root: Tensor,
    start_rpy: Tensor,
    start_joint: Tensor,
    root_target: Tensor,
    rpy_target: Tensor,
    candidate_joint: Tensor,
    command_axis: Tensor,
    crossing_mask: Tensor,
    crossing_offset_w: Tensor,
    root_lift_mask: Tensor,
    duration_steps: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    continuation_mask: Tensor | None = None,
    continuation_lift_w: Tensor | None = None,
    continuation_start_tau: Tensor | None = None,
    samples_override: int | None = None,
    root_path: Tensor | None = None,
    rpy_path: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor, Tensor, Tensor]:
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
    crossing_lift = root_lift_mask[..., None].to(root_target.dtype) * float(
        cfg.nominal.small_cross_root_lift_m
    )
    end_root = root_target[:, :, None].expand(-1, -1, candidates, -1)
    end_root = torch.cat(
        (end_root[..., :2], end_root[..., 2:3] + crossing_lift), dim=-1
    )
    end_state = torch.cat(
        (
            end_root,
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
    smooth = 10.0 * fraction.pow(3) - 15.0 * fraction.pow(4) + 6.0 * fraction.pow(5)
    if root_path is None:
        root = (
            root_start
            + fraction * (root_target[:, :, None, None] - root_start)
        ).expand(-1, -1, candidates, -1, -1)
    else:
        if rpy_path is None or root_path.shape != (batch, legs, 13, 3):
            raise ValueError("preview root/rpy paths must have shape [B,4,13,3]")
        sample_node = fraction[..., 0] * float(root_path.shape[-2] - 1)
        lower_node = sample_node.floor().to(torch.long)
        upper_node = (lower_node + 1).clamp_max(int(root_path.shape[-2] - 1))
        blend = (sample_node - lower_node.to(sample_node.dtype))[..., None]

        def interpolate(path: Tensor) -> Tensor:
            source = path[:, :, None].expand(-1, -1, candidates, -1, -1)
            lower_index = lower_node[..., None].expand(
                batch, legs, candidates, -1, 3
            )
            upper_index = upper_node[..., None].expand_as(lower_index)
            lower_value = torch.gather(source, 3, lower_index)
            upper_value = torch.gather(source, 3, upper_index)
            return lower_value + blend * (upper_value - lower_value)

        root = interpolate(root_path)
    root = root + crossing_root_lift_offset(root_lift_mask, fraction, cfg)
    rpy_start = start_rpy[:, :, None, None]
    rpy_delta = rpy_target[:, :, None, None] - rpy_start
    yaw_delta = torch.remainder(rpy_delta[..., 2:3] + torch.pi, 2.0 * torch.pi) - torch.pi
    rpy_delta = torch.cat((rpy_delta[..., :2], yaw_delta), dim=-1)
    if rpy_path is None:
        rpy = (rpy_start + fraction * rpy_delta).expand(
            -1, -1, candidates, -1, -1
        )
    else:
        rpy = interpolate(rpy_path)

    start_geometry = go2_collision_geometry(
        start_root, start_rpy, start_joint
    )
    lift_index = torch.arange(legs, device=start.device).view(1, legs, 1, 1)
    lift = torch.gather(
        start_geometry.foot_center_w,
        2,
        lift_index.expand(batch, legs, 1, 3),
    ).squeeze(2)[:, :, None, None]
    tau = fraction
    if continuation_mask is not None:
        if continuation_lift_w is None or continuation_start_tau is None:
            raise ValueError(
                "continued sweep requires lift point and absolute start tau"
            )
        continued = continuation_mask[..., None, None]
        lift = torch.where(
            continued,
            continuation_lift_w[:, :, None, None],
            lift,
        )
        tau0 = torch.where(
            continuation_mask,
            continuation_start_tau[:, :, None].expand_as(continuation_mask),
            torch.zeros_like(continuation_mask, dtype=start.dtype),
        )
        tau = tau0[..., None, None] + fraction * (
            1.0 - tau0[..., None, None]
        )
    endpoint_geometry = go2_collision_geometry(
        end_state[..., :3], end_state[..., 3:6], end_state[..., 6:]
    )
    endpoint_index = torch.arange(legs, device=start_root.device).view(1, legs, 1, 1, 1)
    endpoint_index = endpoint_index.expand(batch, legs, candidates, 1, 3)
    target = torch.gather(
        endpoint_geometry.foot_center_w, 3, endpoint_index
    ).squeeze(3)[:, :, :, None]

    bump = 64.0 * tau.pow(3) * (1.0 - tau).pow(3)
    side = constant_like(start, "selector_leg_side", LEG_SIDE_SIGNS).view(1, legs, 1, 1)
    endpoint_yaw = rpy_target[..., 2][:, :, None, None]
    leg_outward = torch.stack(
        (-torch.sin(endpoint_yaw) * side, torch.cos(endpoint_yaw) * side), dim=-1
    )
    outward = torch.where(
        crossing_mask[..., None, None],
        crossing_offset_w[..., None, :],
        float(cfg.nominal.swing_outward_offset_m) * leg_outward,
    )
    foot_xy = swing_xy_profile(
        lift[..., :2],
        target[..., :2],
        command_axis[:, :, None, None],
        tau,
        crossing=crossing_mask[..., None, None],
        outward=outward,
        cfg=cfg,
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
    apex_margin = torch.where(
        crossing_mask,
        apex_height.new_full((), float(cfg.nominal.small_cross_apex_margin_m)),
        apex_height.new_full((), float(cfg.nominal.swing_apex_margin_m)),
    )
    apex_z = (
        apex_height.amax(dim=-1)
        + float(cfg.terrain.foot_radius_m)
        + apex_margin
    )[..., None, None]
    apex_z = torch.maximum(
        apex_z,
        torch.maximum(lift[..., 2:3], target[..., 2:3]),
    )
    foot_z = swing_height_profile(
        lift[..., 2:3],
        apex_z,
        target[..., 2:3],
        tau,
        cfg,
    )
    foot_path = torch.cat((foot_xy, foot_z), dim=-1)
    if root.shape != rpy.shape or root.shape != foot_path.shape:
        raise ValueError(
            f"selector sweep pose/path shapes differ: root={tuple(root.shape)}, "
            f"rpy={tuple(rpy.shape)}, foot={tuple(foot_path.shape)}"
        )

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
    joint_margin_by_axis = torch.minimum(
        selected_joint - lower,
        upper - selected_joint,
    ).amin(dim=-2)
    haa_margin_by_sample = torch.minimum(
        selected_joint[..., 0] - lower[..., 0],
        upper[..., 0] - selected_joint[..., 0],
    )
    haa_index = haa_margin_by_sample.argmin(dim=-1)
    haa_gather = haa_index[..., None, None].expand(-1, -1, -1, 1, 3)
    haa_foot = torch.gather(foot_path, 3, haa_gather).squeeze(3)
    haa_tau = haa_index.to(foot_path.dtype) / float(samples - 1)
    haa_debug = torch.stack((haa_tau, haa_foot[..., 1], haa_foot[..., 2]), dim=-1)
    joint_safe_by_axis = (
        (selected_joint >= lower + float(cfg.touchdown.joint_margin_rad))
        & (selected_joint <= upper - float(cfg.touchdown.joint_margin_rad))
    ).all(dim=-2)
    joint_safe = joint_safe_by_axis.all(dim=-1)
    gait_node = constant_like(
        selected_joint,
        "selector_sweep_gait_node",
        tuple(float(index) for index in range(int(cfg.gait.swing_steps) + 1)),
    ).view(1, 1, 1, -1)
    duration = duration_steps.clamp(1, int(cfg.gait.swing_steps))
    gait_sample_index = torch.round(
        gait_node
        * float(samples - 1)
        / duration[:, :, None, None].to(selected_joint.dtype)
    ).to(torch.long).clamp(0, samples - 1)
    gait_joint = torch.gather(
        selected_joint,
        3,
        gait_sample_index[..., None].expand(-1, -1, candidates, -1, 3),
    )
    gait_root = torch.gather(
        root,
        3,
        gait_sample_index[..., None].expand(-1, -1, candidates, -1, 3),
    )
    gait_rpy = torch.gather(
        rpy,
        3,
        gait_sample_index[..., None].expand(-1, -1, candidates, -1, 3),
    )
    gait_reachable = torch.gather(
        selected_reachable,
        3,
        gait_sample_index.expand(-1, -1, candidates, -1),
    ).all(dim=-1)
    gait_joint_safe_by_axis = (
        (gait_joint >= lower[..., None, :][..., :3] + float(cfg.touchdown.joint_margin_rad))
        & (gait_joint <= upper[..., None, :][..., :3] - float(cfg.touchdown.joint_margin_rad))
    )
    gait_joint_safe = gait_joint_safe_by_axis.all(dim=(-1, -2))
    gait_edge_active = gait_node[..., 1:] <= duration[:, :, None, None]
    maximum_joint_step = float(cfg.solver.joint_velocity_limit) * float(
        cfg.runtime.dt
    )
    joint_rate_step = torch.where(
        gait_edge_active[..., None],
        (gait_joint[..., 1:, :] - gait_joint[..., :-1, :]).abs(),
        torch.zeros_like(gait_joint[..., 1:, :]),
    )
    joint_rate_margin_by_axis = maximum_joint_step - joint_rate_step.amax(dim=-2)
    joint_rate_safe = (joint_rate_margin_by_axis >= 0.0).all(dim=-1)
    # At a one-node touchdown, the warm gait-rate preview is not the path
    # that nominal publishes: x0 is injected from measurement and x1 is
    # checked by the exact nominal and line-search contracts below.
    joint_rate_safe = torch.where(
        duration_steps[..., None] <= 1,
        torch.ones_like(joint_rate_safe),
        joint_rate_safe,
    )
    subdivisions = int(cfg.terrain.sweep_subdivisions)
    joint_linear_leg = leg_index[..., :1].expand(
        batch, legs, candidates, int(cfg.gait.swing_steps)
    )

    def joint_linear_part_safe(points: Tensor, channel: int, vertical: float) -> Tensor:
        points_per_edge = int(
            points.numel()
            // (batch * legs * candidates * int(cfg.gait.swing_steps) * 3)
        )
        flattened = points.reshape(
            batch, legs * candidates * int(cfg.gait.swing_steps) * points_per_edge, 3
        )
        inflated_height, valid_query = query_inflated_height_world(
            field, flattened, channel=channel
        )
        clearance = flattened[..., 2] - inflated_height - float(vertical)
        safe_by_edge = (valid_query & (clearance >= 0.0)).reshape(
            batch,
            legs,
            candidates,
            int(cfg.gait.swing_steps),
            points_per_edge,
        ).all(dim=-1)
        return torch.where(
            gait_edge_active.expand(-1, -1, candidates, -1),
            safe_by_edge,
            torch.ones_like(safe_by_edge),
        ).all(dim=-1)

    def joint_linear_capsule_safe(
        endpoints: Tensor, channel: int, vertical: float
    ) -> Tensor:
        capsule_fraction = constant_like(
            endpoints,
            f"selector_joint_linear_capsule_fraction_{capsule_samples}",
            tuple(
                index / float(capsule_samples - 1)
                for index in range(capsule_samples)
            ),
        ).view(1, 1, 1, 1, capsule_samples, 1)
        points = endpoints[..., :1, :] + capsule_fraction * (
            endpoints[..., 1:, :] - endpoints[..., :1, :]
        )
        return joint_linear_part_safe(points, channel, vertical)

    joint_linear_components = {
        name: torch.ones(
            (batch, legs, candidates), dtype=torch.bool, device=gait_joint.device
        )
        for name in (
            "sweep_joint_linear_foot",
            "sweep_joint_linear_knee",
            "sweep_joint_linear_calf",
            "sweep_joint_linear_thigh",
        )
    }
    for subdivision in range(subdivisions + 1):
        fraction = subdivision / float(subdivisions)

        def interpolate_edge(value: Tensor) -> Tensor:
            return value[..., :-1, :] + fraction * (
                value[..., 1:, :] - value[..., :-1, :]
            )

        joint_linear_geometry = go2_selected_leg_collision_geometry(
            interpolate_edge(gait_root),
            interpolate_edge(gait_rpy),
            interpolate_edge(gait_joint),
            joint_linear_leg,
        )
        sampled_components = {
            "sweep_joint_linear_foot": joint_linear_part_safe(
                joint_linear_geometry.foot_center_w,
                0,
                terrain.foot_radius_m,
            ),
            "sweep_joint_linear_knee": joint_linear_part_safe(
                joint_linear_geometry.knee_center_w,
                1,
                terrain.knee_radius_m + terrain.link_margin_m,
            ),
            "sweep_joint_linear_calf": joint_linear_capsule_safe(
                joint_linear_geometry.calf_endpoints_w,
                2,
                terrain.calf_radius_m + terrain.link_margin_m,
            ),
            "sweep_joint_linear_thigh": joint_linear_capsule_safe(
                joint_linear_geometry.thigh_endpoints_w,
                3,
                terrain.thigh_radius_m + terrain.link_margin_m,
            ),
        }
        joint_linear_components = {
            name: joint_linear_components[name] & sampled_components[name]
            for name in joint_linear_components
        }
    resolved_components = {
        "sweep_resolved_foot": segment_resolved(foot),
        "sweep_resolved_knee": segment_resolved(knee),
        "sweep_resolved_calf": segment_resolved(calf),
        "sweep_resolved_thigh": segment_resolved(thigh),
    }
    resolved = torch.stack(tuple(resolved_components.values()), dim=-1).all(dim=-1)
    components = {
        "sweep_apex_valid": apex_valid.all(dim=-1),
        # The published preview is the 13-node IK path plus the exact
        # joint-linear interval sweep below.  Continuous IK samples are
        # retained as diagnostics, but cannot reject a different trajectory.
        "sweep_reachable": gait_reachable,
        "sweep_joint_haa": gait_joint_safe_by_axis[..., 0].all(dim=-1),
        "sweep_joint_hfe": gait_joint_safe_by_axis[..., 1].all(dim=-1),
        "sweep_joint_kfe": gait_joint_safe_by_axis[..., 2].all(dim=-1),
        "sweep_joint": gait_joint_safe,
        "sweep_joint_rate": joint_rate_safe,
        "sweep_foot": joint_linear_components["sweep_joint_linear_foot"],
        "sweep_knee": joint_linear_components["sweep_joint_linear_knee"],
        "sweep_calf": joint_linear_components["sweep_joint_linear_calf"],
        "sweep_thigh": joint_linear_components["sweep_joint_linear_thigh"],
        **joint_linear_components,
        **resolved_components,
    }
    safe = torch.stack(tuple(components.values()), dim=-1).all(dim=-1)
    return (
        safe,
        resolved,
        components,
        joint_margin_by_axis,
        joint_rate_margin_by_axis,
        haa_debug,
    )


def _candidate_support_safe(
    warm: Tensor,
    candidate_w: Tensor,
    event_step: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    root_lift_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
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
    if root_lift_mask is not None:
        support_tau = offsets.view(1, 1, 1, -1).to(root.dtype) / float(
            cfg.gait.stance_steps
        )
        lift = (
            root_lift_mask[..., None, None].to(root.dtype)
            * (1.0 - support_tau[..., None])
            * float(cfg.nominal.small_cross_root_lift_m)
        )
        root = torch.cat((root[..., :2], root[..., 2:3] + lift), dim=-1)
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
    geometry = go2_selected_leg_collision_geometry(root, rpy, joint, leg_index)
    capsule_samples = int(cfg.terrain.capsule_samples)
    capsule_fraction = constant_like(
        warm,
        f"selector_support_capsule_fraction_{capsule_samples}",
        tuple(index / float(capsule_samples - 1) for index in range(capsule_samples)),
    ).view(1, 1, 1, 1, capsule_samples, 1)

    def sample_capsule(endpoints: Tensor) -> Tensor:
        return endpoints[..., :1, :] + capsule_fraction * (
            endpoints[..., 1:, :] - endpoints[..., :1, :]
        )

    def part_safe(points: Tensor, channel: int, vertical_margin: float) -> Tensor:
        points_per_candidate = int(points.numel() // (batch * legs * candidates * 3))
        flattened = points.reshape(batch, legs * candidates * points_per_candidate, 3)
        inflated_height, valid = query_inflated_height_world(
            field, flattened, channel=channel
        )
        clearance = flattened[..., 2] - inflated_height - float(vertical_margin)
        return (valid & (clearance >= 0.0)).reshape(
            batch, legs, candidates, points_per_candidate
        ).all(dim=-1)

    terrain = cfg.terrain
    components = {
        "support_reachable": reachable.all(dim=-1),
        "support_joint": joint_safe,
        "support_foot": part_safe(
            geometry.foot_center_w, 0, terrain.foot_radius_m
        ),
        "support_knee": part_safe(
            geometry.knee_center_w,
            1,
            terrain.knee_radius_m + terrain.link_margin_m,
        ),
        "support_calf": part_safe(
            sample_capsule(geometry.calf_endpoints_w),
            2,
            terrain.calf_radius_m + terrain.link_margin_m,
        ),
        "support_thigh": part_safe(
            sample_capsule(geometry.thigh_endpoints_w),
            3,
            terrain.thigh_radius_m + terrain.link_margin_m,
        ),
    }
    safe = torch.stack(tuple(components.values()), dim=-1).all(dim=-1)
    return safe, components


def select_touchdowns(
    measured: JointMpcRtiState,
    command_body: Tensor,
    schedule: FixedTrotSchedule,
    warm_nodes: Tensor,
    field: JointMpcPerceptiveField,
    cfg: JointMpcRtiCfg,
    *,
    previous_plan: TouchdownPlan | None = None,
    previous_target_w: Tensor | None = None,
    previous_selected_index: Tensor | None = None,
    previous_crossing: Tensor | None = None,
    previous_remaining_steps: Tensor | None = None,
    previous_swing_offset_w: Tensor | None = None,
    previous_lift_w: Tensor | None = None,
    terrain_field: JointMpcTerrainField | None = None,
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

    base_offset = constant_like(
        warm,
        "touchdown_candidate_xy",
        tuple(
            (x_value, y_value)
            for x_value in cfg.touchdown.candidate_x_m
            for y_value in cfg.touchdown.candidate_y_m
        ),
    )
    small_present = field.small_mask.any(dim=(1, 2))
    outer_x = base_offset[:, 0].abs() == max(abs(value) for value in cfg.touchdown.candidate_x_m)
    outer_y = base_offset[:, 1].abs() == max(abs(value) for value in cfg.touchdown.candidate_y_m)
    expanded_offset = base_offset[None, None].expand(batch, 4, -1, -1)
    crossing_inner = float(cfg.touchdown.small_cross_candidate_inner_m)
    crossing_extent = float(cfg.touchdown.small_cross_candidate_extent_m)
    translation = command[:, :2].abs()
    translation_active = translation.amax(dim=-1) > 1.0e-6
    extend_x = small_present & translation_active & (translation[:, 0] >= translation[:, 1])
    extend_y = small_present & translation_active & (translation[:, 1] > translation[:, 0])
    crossing_offset = torch.stack(
        (
            torch.where(
                extend_x[:, None, None] & outer_x[None, None],
                torch.sign(expanded_offset[..., 0]) * crossing_extent,
                torch.where(
                    extend_x[:, None, None] & (expanded_offset[..., 0] != 0.0),
                    torch.sign(expanded_offset[..., 0]) * crossing_inner,
                    expanded_offset[..., 0],
                ),
            ),
            torch.where(
                extend_y[:, None, None] & outer_y[None, None],
                torch.sign(expanded_offset[..., 1]) * crossing_extent,
                torch.where(
                    extend_y[:, None, None] & (expanded_offset[..., 1] != 0.0),
                    torch.sign(expanded_offset[..., 1]) * crossing_inner,
                    expanded_offset[..., 1],
                ),
            ),
        ),
        dim=-1,
    )
    offset = crossing_offset
    rotated_offset = torch.stack(
        (
            cosine[..., None] * offset[..., 0]
            - sine[..., None] * offset[..., 1],
            sine[..., None] * offset[..., 0]
            + cosine[..., None] * offset[..., 1],
        ),
        dim=-1,
    )
    warm_geometry = go2_collision_geometry(
        warm[..., :3], warm[..., 3:6], warm[..., 6:]
    )
    warm_target = _gather_nodes(warm_geometry.foot_center_w, event_step)
    command_axis, command_speed = _world_command_axis(
        command[:, None].expand(-1, 4, -1), yaw
    )
    lateral_axis = torch.stack((-command_axis[..., 1], command_axis[..., 0]), dim=-1)
    warm_lateral = (
        (warm_target[..., :2] - hip_xy) * lateral_axis
    ).sum(dim=-1, keepdim=True)
    small_center = hip_xy + warm_lateral * lateral_axis
    candidate_center = torch.where(
        small_present[:, None, None], small_center, hip_xy
    )
    candidate_xy = candidate_center[:, :, None] + rotated_offset
    measured_geometry = go2_collision_geometry(
        measured.root_pos_w, measured.root_rpy_w, measured.joint_pos
    )
    lift = measured_geometry.foot_center_w
    lane_hold_slot = constant_like(
        warm,
        "touchdown_lane_hold_slot",
        tuple(float(index == 14) for index in range(25)),
    ).to(torch.bool).view(1, 1, 25)
    hold_seed = candidate_xy[:, :, 14]
    hold_progress = (
        (hold_seed - lift[..., :2]) * command_axis
    ).sum(dim=-1, keepdim=True)
    lane_hold = lift[..., :2] + hold_progress * command_axis
    candidate_xy = torch.where(
        (small_present[:, None, None] & lane_hold_slot)[..., None],
        lane_hold[:, :, None],
        candidate_xy,
    )
    candidate_index = constant_like(
        candidate_xy,
        "touchdown_candidate_index",
        tuple(float(value) for value in range(25)),
    ).view(1, 1, 25)
    current_swing = schedule.swing[:, 0]
    continued_crossing = torch.zeros(
        batch, 4, dtype=torch.bool, device=warm.device
    )
    ordinary_continuation = torch.zeros_like(continued_crossing)
    continued_candidate = torch.zeros(
        batch, 4, 25, dtype=torch.bool, device=warm.device
    )
    inject_previous = torch.zeros_like(continued_candidate)
    if previous_plan is not None or previous_target_w is not None:
        if previous_plan is not None:
            prior_target = previous_plan.target_w
            prior_index = previous_plan.selected_index
            prior_valid = previous_plan.valid
        else:
            assert previous_target_w is not None
            assert previous_selected_index is not None
            prior_target = previous_target_w
            prior_index = previous_selected_index
            prior_valid = torch.ones(batch, 4, dtype=torch.bool, device=warm.device)
        if terrain_field is not None:
            prior_xy = prior_target[..., :2].unsqueeze(2)
            _, _, _, prior_obstacle_out = _small_crossing_offsets(
                lift[..., :2],
                prior_xy,
                command_axis,
                terrain_field,
                cfg,
            )
            prior_progress = (
                (prior_target[..., :2] - lift[..., :2]) * command_axis
            ).sum(dim=-1)
            ordinary_continuation = _post_obstacle_continuation(
                current_swing=current_swing,
                candidate_progress=prior_progress,
                obstacle_out=prior_obstacle_out[..., 0],
                margin_m=float(cfg.touchdown.landing_after_margin_m),
            ) & prior_valid
        continuation_window = (
            schedule.swing[:, 0]
            if previous_remaining_steps is None
            else previous_remaining_steps > 0
        )
        crossing_continuation = continuation_window & (
            torch.zeros(batch, 4, dtype=torch.bool, device=warm.device)
            if previous_crossing is None
            else previous_crossing
        )
        target_continuation = (
            latched | crossing_continuation | ordinary_continuation
        )
        exact_previous = (
            target_continuation[..., None]
            & prior_valid[..., None]
            & (candidate_index == prior_index[..., None])
        )
        continued_crossing = crossing_continuation
        (
            candidate_xy,
            continued_candidate,
            exact_crossing,
        ) = _continuation_retarget_candidates(
            candidate_xy,
            prior_target[..., :2],
            prior_index,
            continued_crossing & prior_valid,
            current_swing=current_swing,
            event_yaw=yaw,
            cfg=cfg,
        )
        inject_previous = exact_previous | continued_candidate
        candidate_xy = torch.where(
            exact_previous[..., None] & ~continued_candidate[..., None],
            prior_target[:, :, None, :2],
            candidate_xy,
        )
    if terrain_field is None:
        candidate_swing_offset = torch.zeros_like(candidate_xy)
        candidate_cross_opportunity = torch.zeros_like(
            candidate_xy[..., 0], dtype=torch.bool
        )
        candidate_obstacle_out = torch.full_like(
            candidate_xy[..., 0], -torch.inf
        )
        candidate_obstacle_in = torch.full_like(
            candidate_xy[..., 0], torch.inf
        )
    else:
        (
            candidate_swing_offset,
            candidate_cross_opportunity,
            candidate_obstacle_in,
            candidate_obstacle_out,
        ) = (
            _small_crossing_offsets(
                lift[..., :2], candidate_xy, command_axis, terrain_field, cfg
            )
        )
    if previous_swing_offset_w is not None:
        prior_swing_offset = torch.as_tensor(
            previous_swing_offset_w,
            dtype=candidate_xy.dtype,
            device=candidate_xy.device,
        )
        if prior_swing_offset.shape != (batch, 4, 2):
            raise ValueError("previous_swing_offset_w must have shape [B,4,2]")
        candidate_swing_offset = torch.where(
            (exact_previous | exact_crossing)[..., None],
            prior_swing_offset[:, :, None],
            candidate_swing_offset,
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

    corridor_samples = int(cfg.touchdown.corridor_samples)
    corridor_fraction = constant_like(
        warm,
        f"touchdown_corridor_fraction_{corridor_samples}",
        tuple(index / float(corridor_samples - 1) for index in range(corridor_samples)),
    ).view(1, 1, 1, corridor_samples, 1)
    corridor_xy = swing_xy_profile(
        lift[:, :, None, None, :2],
        candidate_xy[:, :, :, None],
        command_axis[:, :, None, None],
        corridor_fraction,
        crossing=candidate_cross_opportunity[..., None, None],
        outward=candidate_swing_offset[..., None, :],
        cfg=cfg,
    )
    corridor_query = query_perceptive_world(
        field, corridor_xy.reshape(batch, 4 * 25 * corridor_samples, 2)
    )
    straight_corridor_xy = lift[:, :, None, None, :2] + corridor_fraction * (
        candidate_xy[:, :, :, None] - lift[:, :, None, None, :2]
    )
    straight_corridor_query = query_perceptive_world(
        field,
        straight_corridor_xy.reshape(batch, 4 * 25 * corridor_samples, 2),
    )
    small_corridor = corridor_query.small_mask.reshape(batch, 4, 25, corridor_samples)
    large_corridor = straight_corridor_query.large_mask.reshape(
        batch, 4, 25, corridor_samples
    )
    corridor_valid = (
        corridor_query.valid & straight_corridor_query.valid
    ).reshape(batch, 4, 25, corridor_samples).all(dim=-1)
    candidate_progress = (
        (candidate_xy - lift[:, :, None, :2]) * command_axis[:, :, None]
    ).sum(dim=-1)
    staging_safe = _small_staging_mask(
        candidate_progress,
        candidate_obstacle_in,
        candidate_obstacle_out,
        before_margin_m=float(cfg.touchdown.landing_before_margin_m),
        after_margin_m=float(cfg.touchdown.landing_after_margin_m),
        continued_candidate=continued_candidate,
    )
    small_cross_required = (
        small_corridor.any(dim=-1) | candidate_cross_opportunity
    ) & (command_speed[:, :, None] > 1.0e-6)
    crossing_required = small_cross_required | continued_candidate
    new_crossing_mask = crossing_required & ~continued_crossing[..., None]
    crossing_lift_feasible = (
        event_step.to(warm.dtype)[:, :, None]
        * float(cfg.runtime.dt)
        * float(cfg.solver.root_z_velocity_limit)
        >= float(cfg.nominal.small_cross_root_lift_m)
    )
    small_after = _crossing_after_mask(
        lift[..., :2],
        candidate_xy,
        command_axis,
        corridor_xy,
        small_corridor,
        small_cross_required,
        continued_candidate,
        margin_m=float(cfg.touchdown.landing_after_margin_m),
        sdf_obstacle_out=candidate_obstacle_out,
    )
    corridor_safe = corridor_valid & ~large_corridor.any(dim=-1)

    root_ik = root_event[:, :, None].expand(-1, -1, 25, -1)
    root_ik = torch.cat(
        (
            root_ik[..., :2],
            root_ik[..., 2:3]
            + new_crossing_mask[..., None].to(warm.dtype)
            * float(cfg.nominal.small_cross_root_lift_m),
        ),
        dim=-1,
    )
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
    support_safe, support_components = _candidate_support_safe(
        warm,
        candidate_w,
        event_step,
        field,
        cfg,
        root_lift_mask=new_crossing_mask,
    )
    lift_step = (event_step - int(cfg.gait.swing_steps)).clamp_min(0)
    warm_lift_root = _gather_nodes(
        warm[..., :3].unsqueeze(2).expand(-1, -1, 4, -1), lift_step
    )
    warm_lift_rpy = _gather_nodes(
        warm[..., 3:6].unsqueeze(2).expand(-1, -1, 4, -1), lift_step
    )
    warm_lift_joint = _gather_nodes(
        warm[..., 6:].unsqueeze(2).expand(-1, -1, 4, -1), lift_step
    )
    current_swing = schedule.swing[:, 0]
    sweep_start_root = torch.where(
        current_swing[..., None],
        measured.root_pos_w[:, None],
        warm_lift_root,
    )
    sweep_start_rpy = torch.where(
        current_swing[..., None],
        measured.root_rpy_w[:, None],
        warm_lift_rpy,
    )
    sweep_start_joint = torch.where(
        current_swing[..., None],
        measured.joint_pos[:, None],
        warm_lift_joint,
    )
    continuation_mask, continuation_start_tau = _crossing_sweep_continuation(
        phase0,
        continued_candidate,
        swing_steps=int(cfg.gait.swing_steps),
        dtype=warm.dtype,
    )
    primary_path_offset = constant_like(
        warm,
        "selector_primary_sweep_path_offset",
        tuple(float(value) for value in range(int(cfg.gait.swing_steps) + 1)),
    ).to(torch.long)
    primary_liftoff = torch.where(
        current_swing,
        torch.zeros_like(event_step),
        (event_step - int(cfg.gait.swing_steps)).clamp_min(0),
    )
    primary_path_node = (
        primary_liftoff[..., None] + primary_path_offset.view(1, 1, -1)
    ).clamp_max(event_step[..., None])
    primary_root_path = _gather_nodes(
        warm[..., :3].unsqueeze(2).expand(-1, -1, 4, -1), primary_path_node
    )
    primary_rpy_path = _gather_nodes(
        warm[..., 3:6].unsqueeze(2).expand(-1, -1, 4, -1), primary_path_node
    )
    (
        sweep_safe,
        sweep_resolved,
        sweep_components,
        sweep_joint_margin,
        sweep_joint_rate_margin,
        sweep_haa_debug,
    ) = (
        _candidate_leg_sweep_safe(
            sweep_start_root,
            sweep_start_rpy,
            sweep_start_joint,
            root_event,
            rpy_event,
            candidate_joint,
            command_axis,
            crossing_required,
            candidate_swing_offset,
            new_crossing_mask,
            event_step.clamp_max(int(cfg.gait.swing_steps)),
            field,
            cfg,
            continuation_mask=(
                continuation_mask if previous_lift_w is not None else None
            ),
            continuation_lift_w=previous_lift_w,
            continuation_start_tau=(
                continuation_start_tau if previous_lift_w is not None else None
            ),
            root_path=primary_root_path,
            rpy_path=primary_rpy_path,
        )
    )

    available_lead_steps = event_step.clamp_max(int(cfg.gait.stance_steps)).to(
        warm.dtype
    )
    command_target = warm_target[..., :2] + command_axis * (
        command_speed
        * (0.5 * available_lead_steps * float(cfg.runtime.dt))
    )[..., None]
    previous_target = (
        command_target
        if previous_plan is None and previous_target_w is None
        else prior_target[..., :2]
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
        & support_safe
        & staging_safe
        & small_after
        & ((~new_crossing_mask) | crossing_lift_feasible)
        & corridor_safe
        & sweep_safe
    )
    safe = pre_region_safe & candidate_regions.valid
    safe_cross = safe & crossing_required & small_after
    leg_has_cross = safe_cross.any(dim=-1)
    cross_event_priority = torch.where(
        leg_has_cross,
        event_step,
        torch.full_like(event_step, int(cfg.runtime.horizon_steps) + 1),
    )
    selected_cross_leg_index = cross_event_priority.argmin(dim=-1)
    leg_index = torch.arange(4, device=warm.device).view(1, 4)
    selected_cross_leg = (
        leg_index == selected_cross_leg_index[:, None]
    ) & leg_has_cross

    if previous_target_w is not None and previous_selected_index is not None:
        previous_safe = torch.gather(
            safe, 2, previous_selected_index[..., None]
        ).squeeze(-1)
        continued_leg = continued_crossing & previous_safe
        selected_cross_leg = torch.where(
            continued_leg.any(dim=-1, keepdim=True),
            continued_leg,
            selected_cross_leg,
        )
    ordinary_safe = safe & ~crossing_required
    ordinary_or_safe = torch.where(
        ordinary_safe.any(dim=-1, keepdim=True), ordinary_safe, safe
    )
    selection_safe = _committed_crossing_selection(
        safe_cross,
        ordinary_or_safe,
        selected_cross_leg,
        continued_crossing,
        continued_candidate,
    )
    selection_safe = _prefer_current_swing_post_obstacle(
        selection_safe,
        safe,
        candidate_progress,
        candidate_obstacle_in,
        candidate_obstacle_out,
        current_swing=current_swing,
        selected_cross_leg=selected_cross_leg,
        continued_crossing=continued_crossing,
        before_margin_m=float(cfg.touchdown.landing_before_margin_m),
        after_margin_m=float(cfg.touchdown.landing_after_margin_m),
    )
    previous_index = None
    previous_matches = None
    previous_current_safe = None
    if previous_plan is not None or previous_target_w is not None:
        previous_query = query_perceptive_world(
            field, prior_target.reshape(batch, 4, 3)
        )
        previous_safe = (previous_query.valid & previous_query.landing_safe).reshape(batch, 4)
        previous_distance = (
            candidate_w[..., :2] - prior_target[:, :, None, :2]
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
            (latched | continued_crossing)
            & previous_plan.valid
            & previous_safe
            & previous_matches
            & previous_current_safe
            & (latched | continued_crossing | ordinary_continuation)
            & ~selected_cross_leg
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

    (
        preview_start_root,
        preview_start_rpy,
        preview_start_joint,
    ) = _preview_sweep_start(warm, selected, preview_step, cfg)
    preview_root_path, preview_rpy_path = _preview_sweep_pose_path(
        warm, preview_step, cfg
    )

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
            preview_cosine[..., None] * offset[..., 0]
            - preview_sine[..., None] * offset[..., 1],
            preview_sine[..., None] * offset[..., 0]
            + preview_cosine[..., None] * offset[..., 1],
        ),
        dim=-1,
    )
    preview_candidate_xy = preview_hip_xy[:, :, None] + preview_rotated_offset
    preview_hold = candidate_index == selected_index[..., None]
    preview_candidate_xy = torch.where(
        preview_hold[..., None],
        selected[:, :, None, :2],
        preview_candidate_xy,
    )
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

    preview_axis, preview_speed = _world_command_axis(
        command[:, None].expand(-1, 4, -1), preview_yaw
    )
    if terrain_field is None:
        preview_swing_offset = torch.zeros_like(preview_candidate_xy)
        preview_cross_opportunity = torch.zeros_like(
            preview_candidate_xy[..., 0], dtype=torch.bool
        )
        preview_obstacle_out_sdf = torch.full_like(
            preview_candidate_xy[..., 0], -torch.inf
        )
        preview_obstacle_in_sdf = torch.full_like(
            preview_candidate_xy[..., 0], torch.inf
        )
    else:
        (
            preview_swing_offset,
            preview_cross_opportunity,
            preview_obstacle_in_sdf,
            preview_obstacle_out_sdf,
        ) = (
            _small_crossing_offsets(
                selected[..., :2],
                preview_candidate_xy,
                preview_axis,
                terrain_field,
                cfg,
                delayed_profile=False,
            )
        )
    preview_corridor_xy = swing_xy_profile(
        selected[:, :, None, None, :2],
        preview_candidate_xy[:, :, :, None],
        preview_axis[:, :, None, None],
        corridor_fraction,
        crossing=preview_cross_opportunity[..., None, None],
        outward=preview_swing_offset[..., None, :],
        cfg=cfg,
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
    preview_relative = preview_corridor_xy - selected[:, :, None, None, :2]
    preview_projected = (
        preview_relative * preview_axis[:, :, None, None]
    ).sum(dim=-1)
    preview_obstacle_out = torch.where(
        preview_small_corridor,
        preview_projected,
        torch.full_like(preview_projected, -torch.inf),
    ).amax(dim=-1)
    preview_cross_required = (
        preview_small_corridor.any(dim=-1) | preview_cross_opportunity
    ) & (
        preview_speed[:, :, None] > 1.0e-6
    )
    preview_progress = (
        (preview_candidate_xy - selected[:, :, None, :2])
        * preview_axis[:, :, None]
    ).sum(dim=-1)
    preview_obstacle_out = torch.maximum(
        preview_obstacle_out, preview_obstacle_out_sdf
    )
    preview_staging_safe = _small_staging_mask(
        preview_progress,
        preview_obstacle_in_sdf,
        preview_obstacle_out_sdf,
        before_margin_m=float(cfg.touchdown.landing_before_margin_m),
        after_margin_m=float(cfg.touchdown.landing_after_margin_m),
    )
    preview_small_after = (~preview_cross_required) | (
        preview_progress
        >= preview_obstacle_out + float(cfg.touchdown.landing_after_margin_m)
    )
    preview_corridor_safe = preview_corridor_valid & ~preview_large_corridor.any(
        dim=-1
    )
    (
        preview_sweep_safe,
        preview_sweep_resolved,
        preview_sweep_components,
        preview_sweep_joint_margin,
        preview_sweep_joint_rate_margin,
        preview_sweep_haa_debug,
    ) = (
        _candidate_leg_sweep_safe(
            preview_start_root,
            preview_start_rpy,
            preview_start_joint,
            preview_root,
            preview_rpy,
            preview_candidate_joint,
            preview_axis,
            preview_cross_required,
            preview_swing_offset,
            preview_cross_required,
            preview_step.new_full(
                preview_step.shape, int(cfg.gait.swing_steps)
            ),
            field,
            cfg,
            samples_override=int(cfg.touchdown.preview_swing_samples),
            root_path=preview_root_path,
            rpy_path=preview_rpy_path,
        )
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
        & preview_staging_safe
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
    selected_swing_offset = _gather_candidate(
        candidate_swing_offset, selected_index
    )
    preview_selected_swing_offset = _gather_candidate(
        preview_swing_offset, preview_selected_index
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
        small_cross_required=crossing_required,
        small_after_mask=small_after,
        candidate_swing_offset_w=candidate_swing_offset,
        selected_swing_offset_w=selected_swing_offset,
        preview_small_cross_required=preview_cross_required,
        preview_candidate_swing_offset_w=preview_swing_offset,
        preview_selected_swing_offset_w=preview_selected_swing_offset,
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
            "sweep_haa_margin": sweep_joint_margin[..., 0],
            "sweep_hfe_margin": sweep_joint_margin[..., 1],
            "sweep_kfe_margin": sweep_joint_margin[..., 2],
            "sweep_haa_rate_margin": sweep_joint_rate_margin[..., 0],
            "sweep_hfe_rate_margin": sweep_joint_rate_margin[..., 1],
            "sweep_kfe_rate_margin": sweep_joint_rate_margin[..., 2],
            "sweep_haa_tau": sweep_haa_debug[..., 0],
            "sweep_haa_foot_y": sweep_haa_debug[..., 1],
            "sweep_haa_foot_z": sweep_haa_debug[..., 2],
        },
        valid_components={
            "map": map_safe,
            "plane": plane_safe,
            "reachable": reachable,
            "joint": joint_safe,
            **support_components,
            "support": support_safe,
            "staging": staging_safe,
            "small_after": small_after,
            "corridor": corridor_safe,
            **sweep_components,
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
            "staging": preview_staging_safe,
            "small_after": preview_small_after,
            "corridor": preview_corridor_safe,
            **preview_sweep_components,
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
