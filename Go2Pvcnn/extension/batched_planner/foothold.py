"""Batched foothold helpers for the GPU planner."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..convention import yaw_rotation_matrix_batch
from .gait import batched_legs_requiring_touchdown
from .types import GRAVITY, HIP_HEIGHT, HIP_OFFSETS_ARRAY, LEG_SIDE_SIGN, LEG_ORDER


def _precompute_spiral_offsets(search_radius: float, grid_step: float) -> Tensor:
    """Return integer spiral offsets matching raw square-ring enumeration order."""
    if grid_step <= 0.0:
        raise ValueError("grid_step must be positive")

    n_max = max(int(math.floor(float(search_radius) / float(grid_step) + 1e-9)), 0)
    offsets: list[tuple[int, int]] = [(0, 0)]
    for k in range(1, n_max + 1):
        for x in range(-k, k + 1):
            offsets.append((x, -k))
        for y in range(-k + 1, k + 1):
            offsets.append((k, y))
        for x in range(k - 1, -k - 1, -1):
            offsets.append((x, k))
        for y in range(k - 1, -k, -1):
            offsets.append((-k, y))
    return torch.tensor(offsets, dtype=torch.int64)


_LEG_SIDE_SIGNS = torch.tensor(
    [LEG_SIDE_SIGN["FL"], LEG_SIDE_SIGN["FR"], LEG_SIDE_SIGN["RL"], LEG_SIDE_SIGN["RR"]],
    dtype=torch.float64,
)


def _resolve_input_device(*values) -> torch.device:
    devices = [value.device for value in values if isinstance(value, Tensor)]
    if not devices:
        return torch.device("cpu")
    first = devices[0]
    for device in devices[1:]:
        if device != first:
            raise ValueError("batched foothold helpers do not accept tensor inputs on multiple devices")
    return first


def _coerce_tensor(value, *, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64, device=device)


def _as_shape(name: str, value, *, device: torch.device, shape: tuple[int, ...]) -> Tensor:
    tensor = _coerce_tensor(value, device=device)
    if tensor.ndim != len(shape) + 1 or tuple(tensor.shape[1:]) != shape:
        raise ValueError(f"{name} must have shape (N, {', '.join(str(dim) for dim in shape)}); got {tuple(tensor.shape)}")
    return tensor


def _as_batch_vector(name: str, value, *, device: torch.device) -> Tensor:
    tensor = _coerce_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 1:
        return tensor
    raise ValueError(f"{name} must be a scalar or shape (N,); got {tuple(tensor.shape)}")


def _predict_planar_base_xy(base_pos_xy: Tensor, base_yaw: Tensor, vx: Tensor, vy: Tensor, yaw_rate: Tensor, dt: Tensor) -> Tensor:
    small_turn = torch.abs(yaw_rate) < 1e-9
    yaw1 = base_yaw + yaw_rate * dt
    sin0, cos0 = torch.sin(base_yaw), torch.cos(base_yaw)
    sin1, cos1 = torch.sin(yaw1), torch.cos(yaw1)

    dx_small = (vx * cos0 - vy * sin0) * dt
    dy_small = (vx * sin0 + vy * cos0) * dt
    dx_turn = (vx * (sin1 - sin0) + vy * (cos1 - cos0)) / yaw_rate
    dy_turn = (-vx * (cos1 - cos0) + vy * (sin1 - sin0)) / yaw_rate
    dx = torch.where(small_turn, dx_small, dx_turn)
    dy = torch.where(small_turn, dy_small, dy_turn)
    return torch.stack([base_pos_xy[:, 0] + dx, base_pos_xy[:, 1] + dy], dim=-1)


def _raibert_foothold_xy(
    hip_pos_xy: Tensor,
    base_pos_xy: Tensor,
    base_yaw: Tensor,
    ref_vel_xy: Tensor,
    actual_vel_xy: Tensor,
    stance_time: Tensor,
    com_height: Tensor,
    side_sign: Tensor,
    hip_offset: float = 0.05,
) -> Tensor:
    rot = yaw_rotation_matrix_batch(base_yaw).to(dtype=torch.float64)
    hip_rel_world = torch.cat([hip_pos_xy - base_pos_xy, torch.zeros_like(base_yaw[:, None])], dim=-1)
    hip_body = torch.einsum("nji,nj->ni", rot, hip_rel_world)
    hip_body = hip_body.clone()
    hip_body[:, 1] = hip_body[:, 1] + side_sign * float(hip_offset)

    ref_v_body = torch.cat([ref_vel_xy, torch.zeros_like(base_yaw[:, None])], dim=-1)
    delta = 0.5 * stance_time[:, None] * ref_v_body
    lim_v = float(HIP_HEIGHT) * 1.5
    delta = delta.clone()
    delta[:, :2] = torch.clamp(delta[:, :2], -lim_v, lim_v)

    gain = torch.sqrt(com_height / float(GRAVITY))
    err_xy = gain[:, None] * (actual_vel_xy - ref_vel_xy)
    err_xy = torch.clamp(err_xy, -0.05, 0.05)
    error = torch.cat([err_xy, torch.zeros_like(base_yaw[:, None])], dim=-1)

    ref_body = hip_body + delta + error
    ref_world = torch.einsum("nij,nj->ni", rot, ref_body)
    return ref_world[:, :2] + base_pos_xy


def _spiral_search_safe_foothold(
    nominal: Tensor,
    terrain,
    previous: Tensor,
    *,
    search_radius: float,
    grid_step: float,
    max_roughness: float,
    max_step_down: float,
) -> Tensor:
    offsets = _precompute_spiral_offsets(search_radius, grid_step).to(device=nominal.device, dtype=torch.float64)
    offsets_xy = offsets * float(grid_step)
    distance = torch.linalg.norm(offsets_xy, dim=-1)
    within_radius = distance <= float(search_radius) + 1e-9

    if nominal.ndim != 3 or nominal.shape[-1] != 3:
        raise ValueError(f"nominal must have shape (N, L, 3); got {tuple(nominal.shape)}")
    if previous.shape != nominal.shape:
        raise ValueError(f"previous must have shape {tuple(nominal.shape)}; got {tuple(previous.shape)}")

    batch_size, num_legs = nominal.shape[:2]
    candidate_xy = nominal[:, :, None, :2] + offsets_xy[None, None, :, :]
    flat_candidate_xy = candidate_xy.reshape(batch_size, num_legs * offsets_xy.shape[0], 2)
    candidate_z = terrain.height_at(flat_candidate_xy).reshape(batch_size, num_legs, offsets_xy.shape[0])
    roughness = terrain.roughness_at(flat_candidate_xy).reshape(batch_size, num_legs, offsets_xy.shape[0])
    min_allowed_z = previous[..., 2:3] - float(max_step_down)
    valid = within_radius.view(1, 1, -1) & (roughness <= float(max_roughness)) & (candidate_z >= min_allowed_z)

    d_nom = torch.linalg.norm(offsets_xy, dim=-1).view(1, 1, -1)
    d_prev = torch.linalg.norm(candidate_xy - previous[:, :, None, :2], dim=-1)
    score = d_nom + 0.5 * d_prev
    score = torch.where(valid, score, torch.full_like(score, float("inf")))

    best_idx = score.argmin(dim=-1)
    best_score = score.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    best_xy = candidate_xy.gather(-2, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(-2)
    best_z = candidate_z.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    best = torch.cat([best_xy, best_z.unsqueeze(-1)], dim=-1)

    nominal_z = terrain.height_at(nominal[..., :2])
    fallback_nominal = torch.cat([nominal[..., :2], nominal_z.unsqueeze(-1)], dim=-1)
    fallback = torch.where(nominal_z.unsqueeze(-1) >= min_allowed_z, fallback_nominal, previous)
    return torch.where(torch.isfinite(best_score).unsqueeze(-1), best, fallback)


def batched_compute_footholds(
    *,
    base_pos,
    base_yaw,
    base_lin_vel_xy,
    ref_lin_vel_xy,
    hip_positions,
    stance_time,
    com_height,
    terrain,
    previous_footholds,
    touchdown_times,
    yaw_rate=0.0,
    hip_offset: float = 0.05,
    search_radius: float = 0.15,
    search_step: float = 0.03,
    max_step_down: float = float("inf"),
) -> Tensor:
    device = _resolve_input_device(
        base_pos, base_yaw, base_lin_vel_xy, ref_lin_vel_xy, hip_positions, stance_time, com_height, previous_footholds, touchdown_times, yaw_rate
    )
    base_pos_t = _as_shape("base_pos", base_pos, device=device, shape=(3,))
    base_yaw_t = _as_batch_vector("base_yaw", base_yaw, device=device)
    base_lin_vel_xy_t = _as_shape("base_lin_vel_xy", base_lin_vel_xy, device=device, shape=(2,))
    ref_lin_vel_xy_t = _as_shape("ref_lin_vel_xy", ref_lin_vel_xy, device=device, shape=(2,))
    hip_positions_t = _as_shape("hip_positions", hip_positions, device=device, shape=(4, 3))
    stance_time_t = _as_batch_vector("stance_time", stance_time, device=device)
    com_height_t = _as_batch_vector("com_height", com_height, device=device)
    previous_footholds_t = _as_shape("previous_footholds", previous_footholds, device=device, shape=(4, 3))
    touchdown_times_t = _as_shape("touchdown_times", touchdown_times, device=device, shape=(4,))
    yaw_rate_t = _as_batch_vector("yaw_rate", yaw_rate, device=device)

    batch_size = int(base_pos_t.shape[0])
    if not all(
        tensor.shape[0] == batch_size
        for tensor in (
            base_yaw_t,
            base_lin_vel_xy_t,
            ref_lin_vel_xy_t,
            hip_positions_t,
            stance_time_t,
            com_height_t,
            previous_footholds_t,
            touchdown_times_t,
            yaw_rate_t,
        )
    ):
        raise ValueError("all foothold inputs must share batch size")

    lead_dt = touchdown_times_t.reshape(-1)
    lead_dt_flat = lead_dt
    lead_yaw = base_yaw_t.repeat_interleave(4) + yaw_rate_t.repeat_interleave(4) * lead_dt_flat
    lead_base_xy = _predict_planar_base_xy(
        base_pos_t[:, :2].repeat_interleave(4, dim=0),
        base_yaw_t.repeat_interleave(4),
        ref_lin_vel_xy_t[:, 0].repeat_interleave(4),
        ref_lin_vel_xy_t[:, 1].repeat_interleave(4),
        yaw_rate_t.repeat_interleave(4),
        lead_dt_flat,
    )
    lead_base_pos = torch.cat([lead_base_xy, base_pos_t[:, 2:3].repeat_interleave(4, dim=0)], dim=-1)
    lead_rot = yaw_rotation_matrix_batch(lead_yaw).to(dtype=torch.float64)
    hip_offsets = HIP_OFFSETS_ARRAY.to(device=device, dtype=torch.float64)
    hip_offsets_per_row = hip_offsets.repeat(batch_size, 1)
    lead_hips = lead_base_pos + torch.einsum("nij,nj->ni", lead_rot, hip_offsets_per_row)
    side_sign = _LEG_SIDE_SIGNS.to(device=device).repeat(batch_size)

    nominal_xy = _raibert_foothold_xy(
        lead_hips[:, :2],
        lead_base_xy,
        lead_yaw,
        ref_lin_vel_xy_t.repeat_interleave(4, dim=0),
        base_lin_vel_xy_t.repeat_interleave(4, dim=0),
        stance_time_t.repeat_interleave(4),
        com_height_t.repeat_interleave(4),
        side_sign,
        hip_offset=hip_offset,
    )
    nominal_xy = nominal_xy.reshape(batch_size, 4, 2)
    nominal_z = terrain.height_at(nominal_xy)
    nominal = torch.cat([nominal_xy, nominal_z.unsqueeze(-1)], dim=-1)
    best = _spiral_search_safe_foothold(
        nominal,
        terrain,
        previous_footholds_t,
        search_radius=search_radius,
        grid_step=search_step,
        max_roughness=1.0,
        max_step_down=max_step_down,
    )
    return best


def batched_evaluate_touchdowns(
    touchdown_pos,
    liftoff_pos,
    contact_seq,
    touchdown_mask,
    terrain,
    previous_footholds,
    max_reach: float = 0.15,
) -> tuple[Tensor, Tensor, list[str | None]]:
    device = _resolve_input_device(touchdown_pos, liftoff_pos, contact_seq, touchdown_mask, previous_footholds)
    touchdown_pos_t = _as_shape("touchdown_pos", touchdown_pos, device=device, shape=(4, 3))
    liftoff_pos_t = _as_shape("liftoff_pos", liftoff_pos, device=device, shape=(4, 3))
    contact_seq_t = _as_shape("contact_seq", contact_seq, device=device, shape=(contact_seq.shape[1] if isinstance(contact_seq, Tensor) and contact_seq.ndim == 3 else _coerce_tensor(contact_seq, device=device).shape[1], 4))
    touchdown_mask_t = _coerce_tensor(touchdown_mask, device=device).to(dtype=torch.bool)
    if touchdown_mask_t.ndim != 2 or touchdown_mask_t.shape[1] != 4:
        raise ValueError(f"touchdown_mask must have shape (N, 4); got {tuple(touchdown_mask_t.shape)}")
    previous_footholds_t = _as_shape("previous_footholds", previous_footholds, device=device, shape=(4, 3))

    expected_mask = batched_legs_requiring_touchdown(contact_seq_t)
    if not torch.equal(touchdown_mask_t, expected_mask):
        raise ValueError("touchdown_mask must match legs requiring touchdown from contact_seq")

    xy_reach = torch.linalg.norm(touchdown_pos_t[..., :2] - liftoff_pos_t[..., :2], dim=-1)
    active_reach = torch.where(touchdown_mask_t, xy_reach, torch.zeros_like(xy_reach))
    infeasible = torch.any(active_reach > float(max_reach), dim=1)

    roughness = terrain.roughness_at(touchdown_pos_t[..., :2])
    step_down = torch.clamp_min(previous_footholds_t[..., 2] - touchdown_pos_t[..., 2], 0.0)
    travel_xy = torch.linalg.norm(touchdown_pos_t[..., :2] - previous_footholds_t[..., :2], dim=-1)
    score_terms = roughness + 0.5 * step_down + 0.1 * travel_xy + 0.25 * xy_reach
    score = torch.where(touchdown_mask_t, score_terms, torch.zeros_like(score_terms)).sum(dim=1)
    score = torch.where(infeasible, torch.full_like(score, float("inf")), score)
    reasons = ["xy_reach" if bool(flag.item()) else None for flag in infeasible]
    return ~infeasible, score, reasons


def batched_candidate_total_score(original_cmd, candidate_cmd, touchdown_scores, candidate_indices) -> Tensor:
    device = _resolve_input_device(original_cmd, candidate_cmd, touchdown_scores, candidate_indices)
    original_cmd_t = _as_shape("original_cmd", original_cmd, device=device, shape=(3,))
    candidate_cmd_t = _as_shape("candidate_cmd", candidate_cmd, device=device, shape=(3,))
    touchdown_scores_t = _as_batch_vector("touchdown_scores", touchdown_scores, device=device)
    candidate_indices_t = _coerce_tensor(candidate_indices, device=device)
    if candidate_indices_t.ndim == 0:
        candidate_indices_t = candidate_indices_t.reshape(1)
    if candidate_indices_t.ndim != 1:
        raise ValueError(f"candidate_indices must be a scalar or shape (N,); got {tuple(candidate_indices_t.shape)}")
    candidate_indices_t = candidate_indices_t.to(dtype=torch.float64)

    batch_size = int(original_cmd_t.shape[0])
    if candidate_cmd_t.shape[0] != batch_size or touchdown_scores_t.shape[0] != batch_size or candidate_indices_t.shape[0] != batch_size:
        raise ValueError("original_cmd, candidate_cmd, touchdown_scores, and candidate_indices must share batch size")

    delta = candidate_cmd_t - original_cmd_t
    command_delta_penalty = 3.0 * (delta * delta).sum(dim=-1)
    lateral_penalty = 0.25 * torch.abs(candidate_cmd_t[:, 1])
    yaw_penalty = 0.05 * torch.abs(delta[:, 2])
    vy_usage_penalty = torch.where(torch.abs(candidate_cmd_t[:, 1]) > 1e-9, 0.15, 0.0)
    return touchdown_scores_t + command_delta_penalty + lateral_penalty + yaw_penalty + vy_usage_penalty + 1e-4 * candidate_indices_t


__all__ = [
    "_precompute_spiral_offsets",
    "batched_candidate_total_score",
    "batched_compute_footholds",
    "batched_evaluate_touchdowns",
]
