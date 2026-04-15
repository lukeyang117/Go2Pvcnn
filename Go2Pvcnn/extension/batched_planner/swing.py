"""Batched swing target generation for the GPU planner."""

from __future__ import annotations

import torch
from torch import Tensor


def _resolve_input_device(*values) -> torch.device:
    devices = [value.device for value in values if isinstance(value, Tensor)]
    if not devices:
        return torch.device("cpu")
    first = devices[0]
    for device in devices[1:]:
        if device != first:
            raise ValueError("batched swing helpers do not accept tensor inputs on multiple devices")
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


def _hermite_cubic(t: Tensor, p0: Tensor, p1: Tensor, v0: Tensor, v1: Tensor) -> Tensor:
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00[..., None] * p0 + h10[..., None] * v0 + h01[..., None] * p1 + h11[..., None] * v1


def _compute_swing_apex(lift_off: Tensor, touch_down: Tensor, step_height: float, terrain_max_height: Tensor, clearance: float = 0.02) -> Tensor:
    lo_z = lift_off[..., 2]
    td_z = touch_down[..., 2]
    base_z = torch.maximum(lo_z, td_z)
    margin = torch.maximum(
        torch.full_like(base_z, float(step_height)),
        terrain_max_height - torch.minimum(lo_z, td_z) + float(clearance) + float(step_height) * 0.5,
    )
    return base_z + margin


# ═══════════════════════════════════════════════════════════════════════════════
# Task 7: Vectorized swing progress — replaces _leg_swing_progress_and_stance_anchor
# ═══════════════════════════════════════════════════════════════════════════════


def _batched_swing_progress(stance_bool: Tensor) -> tuple[Tensor, Tensor]:
    """Compute swing progress and touchdown-anchor flags for all (N, T, 4) at once.

    Pure tensor ops — no Python for-loops, no .item() calls.

    Returns:
        swing_progress: (N, T, 4) float64, 0..1 within each contiguous swing run
        use_touchdown:  (N, T, 4) bool, True for stance frames after a complete
                        lift-off → landing cycle
    """
    N, T, num_legs = stance_bool.shape
    device = stance_bool.device
    is_swing = ~stance_bool

    if T == 0:
        empty_f = torch.zeros(N, 0, num_legs, dtype=torch.float64, device=device)
        empty_b = torch.zeros(N, 0, num_legs, dtype=torch.bool, device=device)
        return empty_f, empty_b

    # Detect lift-off (stance→swing) and landing (swing→stance) transitions.
    # Frame −1 is treated as stance (prepend True).
    prev_stance = torch.cat([
        torch.ones(N, 1, num_legs, dtype=torch.bool, device=device),
        stance_bool[:, :-1],
    ], dim=1)
    lift_events = prev_stance & is_swing          # (N, T, 4)
    land_events = (~prev_stance) & stance_bool    # (N, T, 4)

    # use_touchdown: stance frames where #lifts == #lands and at least one cycle done
    lifts_cum = torch.cumsum(lift_events.to(torch.int64), dim=1)
    lands_cum = torch.cumsum(land_events.to(torch.int64), dim=1)
    use_touchdown = stance_bool & (lifts_cum == lands_cum) & (lifts_cum > 0)

    # --- idx_in_run via cumsum + cummax forward-fill ---
    # lift_events == swing_starts (first frame of each swing run)
    swing_starts = lift_events

    cum_swing = torch.cumsum(is_swing.to(torch.int64), dim=1)        # monotonic ↑
    offset_at_start = cum_swing * swing_starts.to(torch.int64)       # nonzero only at starts
    offset_filled = offset_at_start.cummax(dim=1).values             # forward-fill
    idx_in_run = (cum_swing - offset_filled) * is_swing.to(torch.int64)  # (N, T, 4)

    # --- run lengths via scatter_add on run_id ---
    run_id = torch.cumsum(swing_starts.to(torch.int64), dim=1) * is_swing.to(torch.int64)
    max_run_id = int(run_id.max().item()) + 1 if run_id.numel() > 0 else 1

    # Flatten to (N*4, T) so scatter_add_ operates per-(batch, leg) along T
    run_id_flat = run_id.permute(0, 2, 1).reshape(N * num_legs, T)
    is_swing_flat = is_swing.to(torch.int64).permute(0, 2, 1).reshape(N * num_legs, T)

    run_counts = torch.zeros(N * num_legs, max_run_id, dtype=torch.int64, device=device)
    run_counts.scatter_add_(1, run_id_flat, is_swing_flat)

    run_len_flat = run_counts.gather(1, run_id_flat)                 # (N*4, T)
    run_len = run_len_flat.reshape(N, num_legs, T).permute(0, 2, 1) # (N, T, 4)

    denom = torch.clamp(run_len - 1, min=1).to(torch.float64)
    swing_progress = torch.where(
        is_swing,
        idx_in_run.to(torch.float64) / denom,
        torch.zeros(1, dtype=torch.float64, device=device),
    )
    return swing_progress, use_touchdown


# ═══════════════════════════════════════════════════════════════════════════════
# Task 8: Branchless swing-phase targets
# ═══════════════════════════════════════════════════════════════════════════════


def _swing_phase_targets(
    swing_progress: Tensor,
    lift_off: Tensor,
    touch_down: Tensor,
    apex_height: Tensor,
) -> Tensor:
    """Hermite-arc foot positions — branchless, fully broadcastable.

    All inputs broadcast over the common shape (typically N, T, 4).
    lift_off / touch_down carry a trailing dim-3; apex_height does not.
    """
    sp = swing_progress[..., None]                                    # (..., 1)
    xy = (1.0 - sp) * lift_off[..., :2] + sp * touch_down[..., :2]   # (..., 2)

    lo_z = lift_off[..., 2]                                           # broadcastable
    td_z = touch_down[..., 2]
    delta = touch_down - lift_off                                     # (..., 3)

    mask_first = swing_progress <= 0.5

    # -- First half (lo → apex) --
    tau_first = swing_progress / 0.5
    zz = torch.zeros_like(lo_z)
    p0f = torch.stack([zz, zz, lo_z], dim=-1)
    p1f = torch.stack([zz, zz, apex_height], dim=-1)
    v0f = torch.zeros_like(p0f)
    v1f = delta
    z_first = _hermite_cubic(tau_first, p0f, p1f, v0f, v1f)[..., 2]

    # -- Second half (apex → td) --
    tau_second = (swing_progress - 0.5) / 0.5
    zz2 = torch.zeros_like(apex_height)
    p0s = torch.stack([zz2, zz2, apex_height], dim=-1)
    p1s = torch.stack([zz, zz, td_z], dim=-1)
    v0s = delta
    v1s = torch.zeros_like(p0s)
    z_second = _hermite_cubic(tau_second, p0s, p1s, v0s, v1s)[..., 2]

    z = torch.where(mask_first, z_first, z_second)
    return torch.cat([xy, z[..., None]], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 9: Vectorized outer loop — no for batch_idx × for leg
# ═══════════════════════════════════════════════════════════════════════════════


def batched_compute_swing_targets(
    contact_seq,
    lift_off_pos,
    touchdown_pos,
    step_height: float,
    terrain_max_heights=None,
    clearance: float = 0.02,
) -> Tensor:
    device = _resolve_input_device(contact_seq, lift_off_pos, touchdown_pos, terrain_max_heights)
    contact_seq_t = _as_shape("contact_seq", contact_seq, device=device, shape=(contact_seq.shape[1] if isinstance(contact_seq, Tensor) and contact_seq.ndim == 3 else _coerce_tensor(contact_seq, device=device).shape[1], 4))
    lift_off_pos_t = _as_shape("lift_off_pos", lift_off_pos, device=device, shape=(4, 3))
    touchdown_pos_t = _as_shape("touchdown_pos", touchdown_pos, device=device, shape=(4, 3))

    batch_size, num_frames = int(contact_seq_t.shape[0]), int(contact_seq_t.shape[1])
    if num_frames == 0:
        return torch.zeros((batch_size, 0, 4, 3), dtype=torch.float64, device=device)

    if terrain_max_heights is None:
        terrain_h = torch.minimum(lift_off_pos_t[..., 2], touchdown_pos_t[..., 2])
    else:
        terrain_h = _as_shape("terrain_max_heights", terrain_max_heights, device=device, shape=(4,))

    # 1. Apex heights — (N, 4)
    apex = _compute_swing_apex(lift_off_pos_t, touchdown_pos_t, float(step_height), terrain_h, clearance=float(clearance))

    # 2. Swing progress — (N, T, 4) each
    stance_bool = contact_seq_t > 0.5
    swing_progress, use_td = _batched_swing_progress(stance_bool)

    # 3. Arc positions — (N, T, 4, 3)
    #    Insert a T=1 broadcast dim so (N, 4, ...) → (N, 1, 4, ...)
    lo = lift_off_pos_t.unsqueeze(1)    # (N, 1, 4, 3)
    td = touchdown_pos_t.unsqueeze(1)   # (N, 1, 4, 3)
    apex_b = apex.unsqueeze(1)          # (N, 1, 4)
    arc = _swing_phase_targets(swing_progress, lo, td, apex_b)  # (N, T, 4, 3)

    # 4. Stance anchor: touchdown after a full cycle, lift-off otherwise
    anchor = torch.where(use_td[..., None], td, lo)  # (N, T, 4, 3)

    # 5. Final blend: stance → anchor, swing → arc
    return torch.where(stance_bool[..., None], anchor, arc)


# ═══════════════════════════════════════════════════════════════════════════════
# Serial reference (kept for debugging / cross-validation)
# ═══════════════════════════════════════════════════════════════════════════════


def _leg_swing_progress_and_stance_anchor(stance: Tensor) -> tuple[Tensor, Tensor]:
    stance_bool = stance.to(dtype=torch.bool)
    n = stance_bool.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.float64, device=stance.device), torch.zeros(0, dtype=torch.bool, device=stance.device)

    prev_stance = torch.cat([torch.ones(1, dtype=torch.bool, device=stance.device), stance_bool[:-1]])
    lift_events = prev_stance & ~stance_bool
    land_events = ~prev_stance & stance_bool

    lifts_cum = torch.cumsum(lift_events.to(torch.int64), dim=0)
    lands_cum = torch.cumsum(land_events.to(torch.int64), dim=0)
    use_touchdown = stance_bool & (lifts_cum == lands_cum) & (lifts_cum > 0)

    is_swing = ~stance_bool
    idxs = torch.arange(n, device=stance.device, dtype=torch.int64)
    swing_starts = is_swing & torch.cat([torch.ones(1, dtype=torch.bool, device=stance.device), ~is_swing[:-1]])

    starts = torch.full((n,), -1, dtype=torch.int64, device=stance.device)
    start_idx = -1
    for i in range(n):
        if bool(swing_starts[i].item()):
            start_idx = int(idxs[i].item())
        starts[i] = start_idx
    idx_in_run = torch.where(is_swing, idxs - starts, torch.zeros_like(idxs))

    swing_end = is_swing & torch.cat([~is_swing[1:], torch.ones(1, dtype=torch.bool, device=stance.device)])
    lengths = torch.zeros(n, dtype=torch.int64, device=stance.device)
    lengths[swing_end] = idx_in_run[swing_end] + 1
    run_len = torch.zeros_like(lengths)
    last_len = 0
    for i in range(n - 1, -1, -1):
        if int(lengths[i].item()) > 0:
            last_len = int(lengths[i].item())
        run_len[i] = last_len

    denom = torch.clamp(run_len - 1, min=1).to(torch.float64)
    swing_progress = torch.zeros(n, dtype=torch.float64, device=stance.device)
    swing_progress[is_swing] = idx_in_run[is_swing].to(torch.float64) / denom[is_swing]
    return swing_progress, use_touchdown


def _swing_phase_targets_serial(swing_progress: Tensor, lift_off: Tensor, touch_down: Tensor, apex_height: Tensor) -> Tensor:
    xy = (1.0 - swing_progress[..., None]) * lift_off[..., :2] + swing_progress[..., None] * touch_down[..., :2]
    lo_z = lift_off[..., 2]
    td_z = touch_down[..., 2]
    delta = touch_down - lift_off
    v_forward = delta

    z = torch.empty_like(lo_z)
    mask_first = swing_progress <= 0.5
    mask_second = ~mask_first

    if torch.any(mask_first):
        tau = swing_progress[mask_first] / 0.5
        p0 = torch.stack([torch.zeros_like(tau), torch.zeros_like(tau), lo_z[mask_first]], dim=-1)
        p1 = torch.stack([torch.zeros_like(tau), torch.zeros_like(tau), apex_height[mask_first]], dim=-1)
        v0 = torch.zeros_like(p0)
        v1 = v_forward[mask_first]
        z[mask_first] = _hermite_cubic(tau, p0, p1, v0, v1)[..., 2]

    if torch.any(mask_second):
        tau = (swing_progress[mask_second] - 0.5) / 0.5
        p0 = torch.stack([torch.zeros_like(tau), torch.zeros_like(tau), apex_height[mask_second]], dim=-1)
        p1 = torch.stack([torch.zeros_like(tau), torch.zeros_like(tau), td_z[mask_second]], dim=-1)
        v0 = v_forward[mask_second]
        v1 = torch.zeros_like(p0)
        z[mask_second] = _hermite_cubic(tau, p0, p1, v0, v1)[..., 2]

    return torch.cat([xy, z[..., None]], dim=-1)


def _batched_compute_swing_targets_serial(
    contact_seq,
    lift_off_pos,
    touchdown_pos,
    step_height: float,
    terrain_max_heights=None,
    clearance: float = 0.02,
) -> Tensor:
    device = _resolve_input_device(contact_seq, lift_off_pos, touchdown_pos, terrain_max_heights)
    contact_seq_t = _as_shape("contact_seq", contact_seq, device=device, shape=(contact_seq.shape[1] if isinstance(contact_seq, Tensor) and contact_seq.ndim == 3 else _coerce_tensor(contact_seq, device=device).shape[1], 4))
    lift_off_pos_t = _as_shape("lift_off_pos", lift_off_pos, device=device, shape=(4, 3))
    touchdown_pos_t = _as_shape("touchdown_pos", touchdown_pos, device=device, shape=(4, 3))

    batch_size, num_frames = int(contact_seq_t.shape[0]), int(contact_seq_t.shape[1])
    if num_frames == 0:
        return torch.zeros((batch_size, 0, 4, 3), dtype=torch.float64, device=device)

    if terrain_max_heights is None:
        terrain_h = torch.minimum(lift_off_pos_t[..., 2], touchdown_pos_t[..., 2])
    else:
        terrain_h = _as_shape("terrain_max_heights", terrain_max_heights, device=device, shape=(4,))

    out = torch.zeros((batch_size, num_frames, 4, 3), dtype=torch.float64, device=device)
    for batch_idx in range(batch_size):
        for leg in range(4):
            lo = lift_off_pos_t[batch_idx, leg]
            td = touchdown_pos_t[batch_idx, leg]
            apex = _compute_swing_apex(lo, td, float(step_height), terrain_h[batch_idx, leg], clearance=float(clearance))
            stance = contact_seq_t[batch_idx, :, leg] > 0.5
            swing_prog, use_td = _leg_swing_progress_and_stance_anchor(stance)
            arc = _swing_phase_targets_serial(
                swing_prog,
                lo.expand(num_frames, -1),
                td.expand(num_frames, -1),
                apex.expand(num_frames),
            )
            anchor = torch.where(use_td[:, None], td.expand(num_frames, -1), lo.expand(num_frames, -1))
            out[batch_idx, :, leg, :] = torch.where(stance[:, None], anchor, arc)
    return out


__all__ = ["batched_compute_swing_targets"]
