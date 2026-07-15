"""Fixed-iteration GPU-friendly distance transforms for semantic grids."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def semantic_mask(semantic_id: Tensor, ids: tuple[int, ...]) -> Tensor:
    mask = torch.zeros_like(semantic_id, dtype=torch.bool)
    for semantic_value in ids:
        mask = torch.logical_or(mask, semantic_id == int(semantic_value))
    return mask


def _shift_seed(seed: Tensor, delta_x: int, delta_y: int) -> Tensor:
    output = torch.full_like(seed, -1)
    nx = int(seed.shape[1])
    ny = int(seed.shape[2])
    src_x0 = max(0, -delta_x)
    src_x1 = min(nx, nx - delta_x)
    src_y0 = max(0, -delta_y)
    src_y1 = min(ny, ny - delta_y)
    dst_x0 = src_x0 + delta_x
    dst_x1 = src_x1 + delta_x
    dst_y0 = src_y0 + delta_y
    dst_y1 = src_y1 + delta_y
    output[:, dst_x0:dst_x1, dst_y0:dst_y1] = seed[:, src_x0:src_x1, src_y0:src_y1]
    return output


def jump_flood_distance(mask: Tensor, *, resolution: float) -> Tensor:
    """Approximate Euclidean distance in metres to the nearest true grid cell."""
    semantic = torch.as_tensor(mask, dtype=torch.bool)
    if semantic.ndim != 3:
        raise ValueError("mask must have shape [B,NX,NY]")
    batch, nx, ny = map(int, semantic.shape)
    x_coord = torch.arange(nx, dtype=torch.long, device=semantic.device).view(1, nx, 1).expand(batch, nx, ny)
    y_coord = torch.arange(ny, dtype=torch.long, device=semantic.device).view(1, 1, ny).expand(batch, nx, ny)
    coord = torch.stack((x_coord, y_coord), dim=-1)
    seed = torch.where(semantic.unsqueeze(-1), coord, torch.full_like(coord, -1))
    max_dim = max(nx, ny)
    jump = 1 << max(0, int(math.ceil(math.log2(max_dim))) - 1)
    while jump >= 1:
        best = seed
        valid_best = best[..., 0] >= 0
        best_delta = (best - coord).to(dtype=torch.float32)
        best_distance_sq = torch.where(
            valid_best,
            (best_delta * best_delta).sum(dim=-1),
            torch.full(valid_best.shape, float("inf"), dtype=torch.float32, device=semantic.device),
        )
        for delta_x, delta_y in (
            (-jump, -jump),
            (-jump, 0),
            (-jump, jump),
            (0, -jump),
            (0, jump),
            (jump, -jump),
            (jump, 0),
            (jump, jump),
        ):
            candidate = _shift_seed(seed, delta_x, delta_y)
            valid_candidate = candidate[..., 0] >= 0
            candidate_delta = (candidate - coord).to(dtype=torch.float32)
            candidate_distance_sq = torch.where(
                valid_candidate,
                (candidate_delta * candidate_delta).sum(dim=-1),
                torch.full(valid_candidate.shape, float("inf"), dtype=torch.float32, device=semantic.device),
            )
            improve = candidate_distance_sq < best_distance_sq
            best = torch.where(improve.unsqueeze(-1), candidate, best)
            best_distance_sq = torch.where(improve, candidate_distance_sq, best_distance_sq)
        seed = best
        jump //= 2
    maximum_distance = float(resolution) * math.sqrt(float((nx - 1) ** 2 + (ny - 1) ** 2))
    distance = torch.sqrt(best_distance_sq) * float(resolution)
    return torch.where(torch.isfinite(distance), distance, distance.new_full((), maximum_distance))


def distance_gradient(distance_m: Tensor, *, resolution: float) -> Tensor:
    """Return local field-frame gradients ordered as d/dx, d/dy."""
    distance = torch.as_tensor(distance_m)
    gradient_x = torch.empty_like(distance)
    gradient_y = torch.empty_like(distance)
    scale = 1.0 / float(resolution)
    gradient_x[:, 1:-1] = 0.5 * scale * (distance[:, 2:] - distance[:, :-2])
    gradient_x[:, 0] = scale * (distance[:, 1] - distance[:, 0])
    gradient_x[:, -1] = scale * (distance[:, -1] - distance[:, -2])
    gradient_y[:, :, 1:-1] = 0.5 * scale * (distance[:, :, 2:] - distance[:, :, :-2])
    gradient_y[:, :, 0] = scale * (distance[:, :, 1] - distance[:, :, 0])
    gradient_y[:, :, -1] = scale * (distance[:, :, -1] - distance[:, :, -2])
    return torch.stack((gradient_x, gradient_y), dim=-1)


__all__ = ["distance_gradient", "jump_flood_distance", "semantic_mask"]
