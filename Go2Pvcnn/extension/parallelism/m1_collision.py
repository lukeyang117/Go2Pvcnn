from __future__ import annotations

import math

import torch
from torch import Tensor

from extension.parallelism.config import OfficialCollisionShapeSpec, ParallelismCfg


def _quat_to_matrix(quat_wxyz: Tensor) -> Tensor:
    quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = quat.unbind(dim=-1)
    two = torch.tensor(2.0, dtype=quat.dtype, device=quat.device)
    return torch.stack(
        (
            torch.stack((1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), dim=-1),
            torch.stack((two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w)), dim=-1),
            torch.stack((two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )


def _box_points(spec: OfficialCollisionShapeSpec, *, dtype, device) -> Tensor:
    half = torch.tensor(spec.size_l, dtype=dtype, device=device) * 0.5
    values = (-1.0, 0.0, 1.0)
    return torch.tensor(
        [(i * half[0], j * half[1], k * half[2]) for i in values for j in values for k in values if (i, j, k) != (0.0, 0.0, 0.0)],
        dtype=dtype,
        device=device,
    )


def _wheel_points(spec: OfficialCollisionShapeSpec, *, dtype, device) -> Tensor:
    half = float(spec.height_m) * 0.5
    radius = float(spec.radius_m)
    points = [
        torch.tensor((-half, 0.0, 0.0), dtype=dtype, device=device),
        torch.tensor((half, 0.0, 0.0), dtype=dtype, device=device),
    ]
    for width in (-half, 0.0, half):
        for index in range(12):
            theta = 2.0 * math.pi * index / 12.0
            points.append(torch.tensor((width, radius * math.cos(theta), radius * math.sin(theta)), dtype=dtype, device=device))
    return torch.stack(points, dim=0)


def build_m1_surface_points_l(
    specs: tuple[OfficialCollisionShapeSpec, ...],
    cfg: ParallelismCfg,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    point_sets = []
    for spec in specs:
        primitive = _box_points(spec, dtype=dtype, device=device) if spec.shape_type == "box" else _wheel_points(spec, dtype=dtype, device=device)
        rotation = _quat_to_matrix(torch.tensor(spec.quat_wxyz_l, dtype=dtype, device=device))
        center = torch.tensor(spec.center_l, dtype=dtype, device=device)
        point_sets.append(torch.matmul(rotation, primitive[..., None]).squeeze(-1) + center)
    padded = torch.zeros(len(specs), 38, 3, dtype=dtype, device=device)
    mask = torch.zeros(len(specs), 38, dtype=torch.bool, device=device)
    for index, points in enumerate(point_sets):
        padded[index, : points.shape[0]] = points
        mask[index, : points.shape[0]] = True
    return padded, mask
