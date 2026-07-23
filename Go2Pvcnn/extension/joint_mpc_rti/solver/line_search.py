"""Five-candidate loss-only line search for direct state trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.tensor_constants import constant_like
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.losses.terrain import effective_foot_surface_height
from extension.joint_mpc_rti.terrain.query import query_world
from extension.joint_mpc_rti.types import JointMpcTerrainField
from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.losses.objective import LossContext
from extension.joint_mpc_rti.model.nominal import NominalTrajectory
from extension.joint_mpc_rti.solver.lq_problem import LqProblem
from extension.joint_mpc_rti.solver.trajectory_qp import JOINT_LOWER, JOINT_UPPER
from extension.joint_mpc_rti.terrain.swept_safety import (
    PART_NAMES,
    evaluate_nodes,
    evaluate_swept_intervals,
)


ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0)
FILTER_NAMES = ("finite", "joint_position", "joint_velocity", "published_kinematics")
HARD_FILTER_NAMES = (
    "fresh_field",
    "finite",
    "joint",
    "root",
    "stance_xyz",
    "touchdown_region",
    "touchdown_plane",
    "node_safety",
    "swept_safety",
    "preview_safety",
    "cross_direction",
)


@dataclass(frozen=True)
class LineSearchResult:
    state: Tensor
    candidates: Tensor
    alphas: Tensor
    candidate_loss: Tensor
    selected_loss: Tensor
    alpha: Tensor
    selected_index: Tensor
    filter_valid: Tensor
    published_swing_safe_z: Tensor
    valid: Tensor
    selected_feasible: Tensor
    used_nominal: Tensor


@dataclass(frozen=True)
class HardSafeLineSearchResult:
    state: Tensor
    candidates: Tensor
    alphas: Tensor
    candidate_loss: Tensor
    selected_loss: Tensor
    alpha: Tensor
    selected_index: Tensor
    alpha_feasible: Tensor
    alpha_reject_bits: Tensor
    minimum_clearance_by_part: Tensor
    publish: Tensor
    stop: Tensor


def _repeat_perceptive_field(field, repeats: int):
    return type(field)(
        **{
            name: (
                value.repeat_interleave(repeats, dim=0)
                if isinstance(value, Tensor) and value.ndim > 0
                else value
            )
            for name, value in vars(field).items()
        }
    )


def _packed_fk(candidates: Tensor) -> Tensor:
    batch, alphas, nodes = candidates.shape[:3]
    flat = candidates.reshape(batch * alphas * nodes, 18)
    return go2_fk(flat[:, :3], flat[:, 3:6], flat[:, 6:]).foot_pos_w.reshape(
        batch, alphas, nodes, 4, 3
    )


def _touchdown_filters(
    candidates: Tensor,
    foot: Tensor,
    nominal: NominalTrajectory,
    context: LossContext,
    cfg: JointMpcRtiCfg,
) -> tuple[Tensor, Tensor]:
    batch, alpha_count, nodes = candidates.shape[:3]
    plan = nominal.perceptive_plan
    if plan is None:
        valid = torch.ones(batch, alpha_count, dtype=torch.bool, device=candidates.device)
        return valid, valid
    node = constant_like(
        candidates, "hard_line_search_node_index", tuple(range(nodes))
    ).to(torch.long).view(1, 1, nodes, 1)
    active = (
        context.schedule.stance[:, None]
        & (node >= plan.event_step[:, None, None])
        & plan.region.valid[:, None, None]
    )
    region_value = torch.einsum(
        "blij,banlj->banli", plan.region.A, foot[..., :2]
    ) + plan.region.b[:, None, None]
    region_ok = ((region_value >= -1.0e-6) | ~active[..., None]).all(
        dim=(2, 3, 4)
    )
    plane = plan.region.plane
    plane_height = plane[:, None, None, :, 0] + torch.einsum(
        "bli,banli->banl",
        plane[..., 1:],
        foot[..., :2] - plan.target_w[:, None, None, :, :2],
    )
    plane_error = (
        foot[..., 2] - plane_height - float(cfg.gait.foot_contact_offset)
    ).abs()
    plane_ok = (
        (plane_error <= float(cfg.terrain.stance_ground_tolerance_m)) | ~active
    ).all(dim=(2, 3))
    return region_ok, plane_ok


def _preview_filter(
    nominal: NominalTrajectory,
    field,
    cfg: JointMpcRtiCfg,
    alpha_count: int,
) -> Tensor:
    if nominal.preview_tail_state is None:
        return torch.ones(
            nominal.state.shape[0], alpha_count, dtype=torch.bool, device=nominal.state.device
        )
    tail = nominal.preview_tail_state
    repeated_tail = tail.repeat_interleave(alpha_count, dim=0)
    repeated_field = _repeat_perceptive_field(field, alpha_count)
    safety = evaluate_swept_intervals(repeated_tail, repeated_field, cfg)
    return safety.safe.all(dim=1).reshape(nominal.state.shape[0], alpha_count)


def hard_safe_line_search(
    nominal: NominalTrajectory,
    direction: Tensor,
    objective: Callable[[Tensor], Tensor],
    context: LossContext,
    problem: LqProblem,
    cfg: JointMpcRtiCfg,
    *,
    expected_refresh_id: Tensor | None = None,
) -> HardSafeLineSearchResult:
    """Evaluate all five nonlinear candidates with the exact publication gates."""
    base = torch.as_tensor(nominal.state)
    delta = torch.as_tensor(direction, dtype=base.dtype, device=base.device)
    if base.shape[1:] != (31, 18) or delta.shape != base.shape:
        raise ValueError("nominal.state and direction must have shape [B,31,18]")
    field = context.perceptive_field
    if field is None:
        raise ValueError("hard-safe line search requires the current perceptive field")
    alphas = constant_like(base, "hard_line_search_alphas", ALPHAS)
    candidates = base[:, None] + alphas[None, :, None, None] * delta[:, None]
    candidates[:, -1] = base
    batch, alpha_count = int(base.shape[0]), len(ALPHAS)
    packed = candidates.reshape(batch * alpha_count, 31, 18)
    candidate_loss = objective(packed).reshape(batch, alpha_count)

    expected = field.refresh_id if expected_refresh_id is None else expected_refresh_id
    fresh = (field.refresh_id == expected)[:, None].expand(-1, alpha_count)
    finite = torch.isfinite(candidates).all(dim=(2, 3)) & torch.isfinite(candidate_loss)
    joint_lower = constant_like(base, "hard_joint_lower", JOINT_LOWER) + float(
        cfg.solver.joint_margin
    )
    joint_upper = constant_like(base, "hard_joint_upper", JOINT_UPPER) - float(
        cfg.solver.joint_margin
    )
    joints = candidates[..., 6:]
    joint_position = ((joints >= joint_lower) & (joints <= joint_upper)).all(dim=(2, 3))
    joint_velocity = (
        (joints[:, :, 1:] - joints[:, :, :-1]).abs()
        <= float(cfg.solver.joint_velocity_limit) * float(cfg.runtime.dt)
    ).all(dim=(2, 3))
    joint_ok = joint_position & joint_velocity
    root = candidates[..., :6]
    support = context.support_height[:, None]
    root_height = root[..., 2] - support
    root_ok = (
        (root_height >= float(cfg.loss_terms.posture_root_clearance) + float(cfg.solver.root_height_min_offset))
        & (root_height <= float(cfg.loss_terms.posture_root_clearance) + float(cfg.solver.root_height_max_offset))
        & (root[..., 3:5].abs() <= float(cfg.solver.root_roll_pitch_limit)).all(dim=-1)
    ).all(dim=2)

    foot = _packed_fk(candidates)
    stance_error = torch.linalg.vector_norm(
        foot - nominal.foot_reference_w[:, None], dim=-1
    )
    stance_ok = (
        (stance_error <= float(cfg.solver.published_stance_tolerance))
        | ~context.schedule.stance[:, None]
    ).all(dim=(2, 3))
    region_ok, plane_ok = _touchdown_filters(candidates, foot, nominal, context, cfg)

    repeated_field = _repeat_perceptive_field(field, alpha_count)
    repeated_contact = context.schedule.stance.repeat_interleave(alpha_count, dim=0)
    node_safety = evaluate_nodes(
        packed, repeated_field, cfg, contact_state=repeated_contact
    )
    swept_safety = evaluate_swept_intervals(
        packed, repeated_field, cfg, contact_state=repeated_contact
    )
    node_ok = node_safety.safe.all(dim=1).reshape(batch, alpha_count)
    swept_ok = swept_safety.safe.all(dim=1).reshape(batch, alpha_count)
    preview_ok = _preview_filter(nominal, field, cfg, alpha_count)

    cross_ok = torch.ones_like(finite)
    plan = nominal.perceptive_plan
    if plan is not None:
        command_xy = context.command_body[:, :2]
        command_norm = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True)
        direction_xy = command_xy / command_norm.clamp_min(1.0e-6)
        root_progress = candidates[:, :, -1, :2] - candidates[:, :, 0, :2]
        forward = torch.einsum("bai,bi->ba", root_progress, direction_xy)
        required = plan.small_cross_required.any(dim=(1, 2))[:, None] & (
            command_norm.squeeze(-1)[:, None] > 1.0e-4
        )
        cross_ok = (forward >= 0.0) | ~required

    filter_ok = torch.stack(
        (
            fresh,
            finite,
            joint_ok,
            root_ok,
            stance_ok,
            region_ok,
            plane_ok,
            node_ok,
            swept_ok,
            preview_ok,
            cross_ok,
        ),
        dim=-1,
    )
    feasible = filter_ok.all(dim=-1)
    selectable = torch.where(
        feasible, candidate_loss, torch.full_like(candidate_loss, torch.inf)
    )
    minimum = selectable.amin(dim=1, keepdim=True)
    tie = selectable <= minimum + float(cfg.solver.line_search_tie_tolerance)
    selected_index = tie.to(torch.int64).argmax(dim=1)
    publish = feasible.any(dim=1)
    selected_index = torch.where(
        publish, selected_index, torch.full_like(selected_index, alpha_count - 1)
    )
    row = constant_like(
        base, f"hard_line_search_batch_index_{batch}", tuple(range(batch))
    ).to(torch.long)
    minimum_clearance = torch.stack(
        tuple(node_safety.minimum_clearance_by_part[name] for name in PART_NAMES), dim=-1
    ).amin(dim=1).reshape(batch, alpha_count, len(PART_NAMES))
    return HardSafeLineSearchResult(
        state=candidates[row, selected_index],
        candidates=candidates,
        alphas=alphas,
        candidate_loss=candidate_loss,
        selected_loss=candidate_loss[row, selected_index],
        alpha=alphas[selected_index],
        selected_index=selected_index,
        alpha_feasible=feasible,
        alpha_reject_bits=~filter_ok,
        minimum_clearance_by_part=minimum_clearance,
        publish=publish,
        stop=~publish,
    )


def parallel_line_search(
    nominal: Tensor,
    direction: Tensor,
    objective: Callable[[Tensor], Tensor],
    *,
    joint_lower: Tensor,
    joint_upper: Tensor,
    joint_velocity_limit: Tensor | float,
    published_stance_anchor_w: Tensor | None = None,
    published_stance_mask: Tensor | None = None,
    published_stance_ground_mask: Tensor | None = None,
    published_stance_tolerance: float = 0.0005,
    published_swing_mask: Tensor | None = None,
    published_terrain_field: JointMpcTerrainField | None = None,
    published_foot_contact_offset: float = 0.022,
    published_swing_clearance_buffer: float = 0.0,
    published_h_wall: float = 0.35,
    dt: float,
    tie_tolerance: float = 1.0e-7,
) -> LineSearchResult:
    """Select the lowest seven-loss candidate after the four approved filters."""
    base = torch.as_tensor(nominal)
    delta = torch.as_tensor(direction, dtype=base.dtype, device=base.device)
    if base.ndim != 3 or base.shape[1:] != (31, 18) or delta.shape != base.shape:
        raise ValueError("nominal and direction must have shape [B,31,18]")
    alphas = constant_like(base, "line_search_alphas", ALPHAS)
    candidates = base[:, None] + alphas[None, :, None, None] * delta[:, None]
    candidates[:, -1] = base
    batch = int(base.shape[0])
    candidate_loss = objective(candidates.reshape(batch * 5, 31, 18)).reshape(batch, 5)

    finite = torch.isfinite(candidates).all(dim=(2, 3)) & torch.isfinite(candidate_loss)
    lower = torch.as_tensor(joint_lower, dtype=base.dtype, device=base.device)
    upper = torch.as_tensor(joint_upper, dtype=base.dtype, device=base.device)
    joints = candidates[..., 6:]
    position_ok = ((joints >= lower) & (joints <= upper)).all(dim=(2, 3))
    velocity_limit = torch.as_tensor(joint_velocity_limit, dtype=base.dtype, device=base.device)
    joint_step = joints[:, :, 1:] - joints[:, :, :-1]
    velocity_ok = (joint_step.abs() <= velocity_limit * float(dt)).all(dim=(2, 3))
    need_stance = published_stance_anchor_w is not None or published_stance_mask is not None
    need_ground = published_stance_ground_mask is not None
    need_swing = published_swing_mask is not None
    need_foot = need_stance or need_ground or need_swing
    foot = candidates.new_zeros(batch, 5, 4, 3)
    if need_foot:
        published = candidates[:, :, 1].reshape(batch * 5, 18)
        foot = go2_fk(
            published[:, :3], published[:, 3:6], published[:, 6:]
        ).foot_pos_w.reshape(batch, 5, 4, 3)
    if not need_stance:
        stance_ok = torch.ones_like(finite)
    elif published_stance_anchor_w is None or published_stance_mask is None:
        raise ValueError("published stance anchor and mask must be provided together")
    else:
        anchor = torch.as_tensor(
            published_stance_anchor_w, dtype=base.dtype, device=base.device
        )
        stance = torch.as_tensor(
            published_stance_mask, dtype=torch.bool, device=base.device
        )
        if anchor.shape != (batch, 4, 3) or stance.shape != (batch, 4):
            raise ValueError("published stance anchor/mask must have shapes [B,4,3]/[B,4]")
        error_xy = torch.linalg.vector_norm(
            foot[..., :2] - anchor[:, None, :, :2], dim=-1
        )
        stance_ok = (
            error_xy <= float(published_stance_tolerance)
        ) | ~stance[:, None]
        stance_ok = stance_ok.all(dim=2)
    need_query = need_ground or need_swing
    if need_query and published_terrain_field is None:
        raise ValueError("published stance ground/swing masks require a terrain field")
    query = None
    if need_query:
        query = query_world(published_terrain_field, foot.reshape(batch * 5, 4, 3))
    if not need_ground:
        ground_ok = torch.ones_like(finite)
    else:
        ground_mask = torch.as_tensor(
            published_stance_ground_mask, dtype=torch.bool, device=base.device
        )
        if ground_mask.shape != (batch, 4):
            raise ValueError("published stance ground mask must have shape [B,4]")
        assert query is not None
        raw_height = query.height_w.reshape(batch, 5, 4)
        ground_error = (
            foot[..., 2] - raw_height - float(published_foot_contact_offset)
        ).abs()
        ground_ok = (
            ground_error <= float(published_stance_tolerance)
        ) | ~ground_mask[:, None]
        ground_ok = ground_ok.all(dim=2)
    safe_z = candidates.new_zeros(batch, 5, 4)
    if not need_swing:
        swing_ok = torch.ones_like(finite)
    else:
        swing = torch.as_tensor(
            published_swing_mask, dtype=torch.bool, device=base.device
        )
        if swing.shape != (batch, 4):
            raise ValueError("published swing mask must have shape [B,4]")
        assert query is not None
        surface = effective_foot_surface_height(
            query.height_w,
            query.small_occupancy,
            query.large_occupancy,
            query.small_propagated_height,
            stance=torch.zeros_like(query.height_w, dtype=torch.bool),
            h_wall=float(published_h_wall),
        ).reshape(batch, 5, 4)
        safe_z = (
            surface
            + float(published_foot_contact_offset)
            + float(published_swing_clearance_buffer)
        )
        swing_ok = (foot[..., 2] >= safe_z - 1.0e-6) | ~swing[:, None]
        swing_ok = swing_ok.all(dim=2)
    published_ok = stance_ok & ground_ok & swing_ok
    filter_valid = torch.stack((finite, position_ok, velocity_ok, published_ok), dim=-1)
    valid = filter_valid.all(dim=-1)

    selectable = torch.where(valid, candidate_loss, torch.full_like(candidate_loss, float("inf")))
    minimum = selectable.amin(dim=1, keepdim=True)
    tie = selectable <= minimum + float(tie_tolerance)
    selected_index = tie.to(torch.int64).argmax(dim=1)
    any_valid = valid.any(dim=1)
    nominal_index = torch.full_like(selected_index, len(ALPHAS) - 1)
    selected_index = torch.where(any_valid, selected_index, nominal_index)
    row = torch.arange(batch, device=base.device)
    state = candidates[row, selected_index]
    selected_loss = candidate_loss[row, selected_index]
    alpha = alphas[selected_index]
    selected_feasible = valid[row, selected_index]
    return LineSearchResult(
        state=state,
        candidates=candidates,
        alphas=alphas,
        candidate_loss=candidate_loss,
        selected_loss=selected_loss,
        alpha=alpha,
        selected_index=selected_index,
        filter_valid=filter_valid,
        published_swing_safe_z=safe_z,
        valid=valid,
        selected_feasible=selected_feasible,
        used_nominal=selected_index == len(ALPHAS) - 1,
    )


__all__ = [
    "ALPHAS",
    "FILTER_NAMES",
    "HARD_FILTER_NAMES",
    "HardSafeLineSearchResult",
    "LineSearchResult",
    "hard_safe_line_search",
    "parallel_line_search",
]
