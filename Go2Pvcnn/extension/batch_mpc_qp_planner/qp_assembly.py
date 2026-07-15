"""Fixed-shape QP matrix assembly for the MPC-QP backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .fields import QpDifferentiableFields
from .gait import QpGaitMasks
from .variables import QpVariableLayout


@dataclass(frozen=True)
class FixedShapeQpMatrices:
    H: Tensor
    g: Tensor
    A: Tensor
    b: Tensor
    E: Tensor
    e: Tensor
    lower: Tensor
    upper: Tensor


def assemble_fixed_shape_qp(
    *,
    fields: QpDifferentiableFields,
    layout: QpVariableLayout,
    gait: QpGaitMasks,
) -> FixedShapeQpMatrices:
    del fields, gait
    batch = int(layout.batch)
    n = int(layout.total_dim)
    device = layout.touchdown_xy_indices.device
    dtype = torch.float32
    eye = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(batch, -1, -1).contiguous()
    H = eye * 1.0e-3
    g = torch.zeros((batch, n), dtype=dtype, device=device)
    ineq_count = int(layout.legs) * 4 + int(layout.horizon) * 2
    eq_count = int(layout.legs) * 2
    A = torch.zeros((batch, ineq_count, n), dtype=dtype, device=device)
    b = torch.zeros((batch, ineq_count), dtype=dtype, device=device)
    E = torch.zeros((batch, eq_count, n), dtype=dtype, device=device)
    e = torch.zeros((batch, eq_count), dtype=dtype, device=device)
    lower = torch.full((batch, n), -0.25, dtype=dtype, device=device)
    upper = torch.full((batch, n), 0.25, dtype=dtype, device=device)
    lower.scatter_(1, layout.semantic_slack.reshape(batch, -1), 0.0)
    lower.scatter_(1, layout.clearance_slack.reshape(batch, -1), 0.0)
    lower.scatter_(1, layout.reachability_slack.reshape(batch, -1), 0.0)
    lower.scatter_(1, layout.stability_slack.reshape(batch, -1), 0.0)
    upper.scatter_(1, layout.semantic_slack.reshape(batch, -1), 10.0)
    upper.scatter_(1, layout.clearance_slack.reshape(batch, -1), 10.0)
    upper.scatter_(1, layout.reachability_slack.reshape(batch, -1), 10.0)
    upper.scatter_(1, layout.stability_slack.reshape(batch, -1), 10.0)
    return FixedShapeQpMatrices(H=H, g=g, A=A, b=b, E=E, e=e, lower=lower, upper=upper)


__all__ = ["FixedShapeQpMatrices", "assemble_fixed_shape_qp"]
