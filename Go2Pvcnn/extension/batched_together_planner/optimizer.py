"""Fixed-shape optimizer placeholders for the together planner core.

The P0 core uses a deterministic rollout. These helpers keep the module
boundary ready for the raw CEM parity pass without dynamic sub-batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import TogetherPlannerConfig


@dataclass(frozen=True)
class TogetherCEMResult:
    params: Tensor
    best_params: Tensor
    best_cost: Tensor
    winner_index: Tensor


def initialize_seed_params(batch_size: int, *, device: torch.device, dtype: torch.dtype, cfg: TogetherPlannerConfig) -> Tensor:
    base = torch.zeros((batch_size, int(cfg.seed_count), 8), device=device, dtype=dtype)
    offsets = torch.linspace(-1.0, 1.0, int(cfg.seed_count), device=device, dtype=dtype).view(1, int(cfg.seed_count), 1)
    basis = torch.linspace(0.02, 0.005, 8, device=device, dtype=dtype).view(1, 1, 8)
    return base + offsets * basis


def refine_cem_two_steps(params: Tensor, costs: Tensor, cfg: TogetherPlannerConfig) -> TogetherCEMResult:
    elite_count = min(int(cfg.elite_count), int(params.shape[1]))
    top_values_0, top_index_0 = torch.topk(costs, k=elite_count, dim=1, largest=False, sorted=True)
    elite_0 = params.gather(1, top_index_0[:, :, None].expand(-1, -1, params.shape[-1]))
    center_0 = elite_0.mean(dim=1, keepdim=True)
    spread_0 = elite_0.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-3)
    offsets_0 = torch.linspace(-1.0, 1.0, params.shape[1], device=params.device, dtype=params.dtype).view(1, params.shape[1], 1)
    basis_0 = torch.linspace(1.0, 0.2, params.shape[-1], device=params.device, dtype=params.dtype).view(1, 1, -1)
    params_1 = center_0 + offsets_0 * spread_0 * basis_0
    top_values_1, top_index_1 = torch.topk(top_values_0, k=1, dim=1, largest=False, sorted=True)
    best_params = params_1.gather(1, top_index_1[:, :, None].expand(-1, -1, params.shape[-1])).squeeze(1)
    return TogetherCEMResult(
        params=params_1,
        best_params=best_params,
        best_cost=top_values_1.squeeze(1),
        winner_index=top_index_0.gather(1, top_index_1).squeeze(1),
    )


__all__ = ["TogetherCEMResult", "initialize_seed_params", "refine_cem_two_steps"]
