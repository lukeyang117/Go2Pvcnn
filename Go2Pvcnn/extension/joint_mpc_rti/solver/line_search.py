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


ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0)
FILTER_NAMES = ("finite", "joint_position", "joint_velocity", "published_kinematics")


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


__all__ = ["ALPHAS", "FILTER_NAMES", "LineSearchResult", "parallel_line_search"]
