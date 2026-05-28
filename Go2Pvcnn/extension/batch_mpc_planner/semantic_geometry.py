"""Semantic geometry helpers for parametric MPC losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LowSmallCircles:
    center_xy: Tensor
    radius: Tensor
    valid: Tensor
    truncated: Tensor


def _world_grid_xy(
    *,
    batch: int,
    rows: int,
    cols: int,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    xs = torch.linspace(float(world_x_range[0]), float(world_x_range[1]), cols, dtype=dtype, device=device)
    ys = torch.linspace(float(world_y_range[0]), float(world_y_range[1]), rows, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, rows * cols, 2).expand(batch, -1, -1)


def low_small_component_circles(
    semantic_map: Tensor,
    *,
    world_x_range: tuple[float, float],
    world_y_range: tuple[float, float],
    max_components: int = 8,
) -> LowSmallCircles:
    sem = torch.as_tensor(semantic_map)
    if sem.ndim == 2:
        sem = sem.unsqueeze(0)
    if sem.ndim != 3:
        raise ValueError(f"semantic_map must be [B,H,W] or [H,W], got {tuple(sem.shape)}")
    batch, rows, cols = int(sem.shape[0]), int(sem.shape[1]), int(sem.shape[2])
    dtype = torch.float32
    device = sem.device
    max_components = max(1, int(max_components))

    low = sem == 1
    labels = torch.arange(rows * cols, dtype=torch.long, device=device).view(1, 1, rows, cols).expand(batch, -1, -1, -1)
    invalid = torch.full_like(labels, rows * cols + 1)
    labels = torch.where(low[:, None, :, :], labels, invalid)
    for _ in range(rows + cols):
        padded = F.pad(labels.to(dtype=torch.float32), (1, 1, 1, 1), value=float(rows * cols + 1)).to(dtype=torch.long)
        neigh = torch.stack(
            (
                padded[:, :, 1:-1, 1:-1],
                padded[:, :, :-2, 1:-1],
                padded[:, :, 2:, 1:-1],
                padded[:, :, 1:-1, :-2],
                padded[:, :, 1:-1, 2:],
            ),
            dim=0,
        )
        labels = torch.where(low[:, None, :, :], neigh.amin(dim=0), invalid)
    labels_flat = labels.reshape(batch, rows * cols)
    low_flat = low.reshape(batch, rows * cols)
    active_labels = torch.where(low_flat, labels_flat, invalid.reshape(batch, rows * cols))
    unique_labels = torch.unique(active_labels, sorted=True)
    unique_labels = unique_labels[unique_labels <= rows * cols]

    center_xy = torch.zeros((batch, max_components, 2), dtype=dtype, device=device)
    radius = torch.zeros((batch, max_components), dtype=dtype, device=device)
    valid = torch.zeros((batch, max_components), dtype=torch.bool, device=device)
    truncated = torch.zeros((batch,), dtype=torch.bool, device=device)
    if int(unique_labels.numel()) == 0:
        return LowSmallCircles(center_xy=center_xy, radius=radius, valid=valid, truncated=truncated)

    grid_xy = _world_grid_xy(
        batch=batch,
        rows=rows,
        cols=cols,
        world_x_range=world_x_range,
        world_y_range=world_y_range,
        dtype=dtype,
        device=device,
    )
    component_labels = unique_labels[:max_components]
    component_mask = labels_flat[:, None, :] == component_labels.view(1, -1, 1)
    component_mask = torch.logical_and(component_mask, low_flat[:, None, :])
    count = component_mask.sum(dim=-1)
    kept = int(component_labels.numel())
    valid[:, :kept] = count > 0
    weights = component_mask.to(dtype=dtype)
    center = (weights[..., None] * grid_xy[:, None, :, :]).sum(dim=2) / count.clamp_min(1).to(dtype=dtype)[..., None]
    dist = torch.linalg.vector_norm(grid_xy[:, None, :, :] - center[:, :, None, :], dim=-1)
    cell_dx = abs(float(world_x_range[1]) - float(world_x_range[0])) / max(cols - 1, 1)
    cell_dy = abs(float(world_y_range[1]) - float(world_y_range[0])) / max(rows - 1, 1)
    cell_margin = 0.5 * (cell_dx * cell_dx + cell_dy * cell_dy) ** 0.5
    comp_radius = torch.where(component_mask, dist, torch.zeros_like(dist)).amax(dim=-1) + float(cell_margin)
    center_xy[:, :kept, :] = center
    radius[:, :kept] = torch.where(valid[:, :kept], comp_radius, torch.zeros_like(comp_radius))
    if int(unique_labels.numel()) > max_components:
        overflow = unique_labels[max_components:]
        overflow_mask = torch.isin(labels_flat, overflow)
        truncated = overflow_mask.any(dim=-1)
    return LowSmallCircles(center_xy=center_xy, radius=radius, valid=valid, truncated=truncated)


__all__ = ["LowSmallCircles", "low_small_component_circles"]
