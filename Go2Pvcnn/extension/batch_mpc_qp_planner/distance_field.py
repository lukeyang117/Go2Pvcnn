"""Fixed-shape semantic distance helpers for the MPC-QP backend."""

from __future__ import annotations

import torch
from torch import Tensor


def fixed_repair_offsets(*, radius_m: float, step_m: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return a deterministic small offset stencil for semantic projection."""

    step = max(float(step_m), 1.0e-4)
    radius = max(float(radius_m), step)
    rings = max(1, int(round(radius / step)))
    base = []
    for r in range(0, rings + 1):
        dist = float(r) * step
        if r == 0:
            base.append((0.0, 0.0))
            continue
        base.extend(
            (
                (dist, 0.0),
                (-dist, 0.0),
                (0.0, dist),
                (0.0, -dist),
                (dist, dist),
                (dist, -dist),
                (-dist, dist),
                (-dist, -dist),
            )
        )
    offsets = torch.tensor(base, dtype=dtype, device=device)
    keep = torch.linalg.vector_norm(offsets, dim=-1) <= radius + 1.0e-6
    return offsets[keep]


def semantic_violation_from_ids(semantic: Tensor) -> Tensor:
    """Return positive violation magnitude for non-ground semantic ids."""

    return (torch.as_tensor(semantic) != 0).to(dtype=torch.float32)


__all__ = ["fixed_repair_offsets", "semantic_violation_from_ids"]
