"""Environment selection helpers for MPC reference rewards."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class MpcTerrainDifficultyPair:
    terrain_cols: tuple[int, ...] | None = None
    terrain_names: tuple[str, ...] | None = None
    terrain_rows: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class MpcReferenceParticipationCfg:
    enabled: bool = True
    exclude_pairs: tuple[MpcTerrainDifficultyPair, ...] = field(default_factory=tuple)
    selection_mode: str = "round_robin"


def _as_1d_long(values, *, device: torch.device, num_envs: int, name: str) -> Tensor:
    out = torch.as_tensor(values, dtype=torch.long, device=device).reshape(-1)
    if int(out.numel()) != int(num_envs):
        raise ValueError(f"{name} must have {num_envs} entries, got {int(out.numel())}")
    return out


def _isin(values: Tensor, allowed: tuple[int, ...] | None) -> Tensor:
    if allowed is None:
        return torch.ones_like(values, dtype=torch.bool)
    if len(allowed) == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    out = torch.zeros_like(values, dtype=torch.bool)
    for item in allowed:
        out = torch.logical_or(out, values == int(item))
    return out


def _name_mask(
    terrain_types: Tensor,
    terrain_names: list[str] | tuple[str, ...] | None,
    names: tuple[str, ...] | None,
    *,
    none_matches: bool,
) -> Tensor:
    if names is None:
        return torch.ones_like(terrain_types, dtype=torch.bool) if none_matches else torch.zeros_like(terrain_types, dtype=torch.bool)
    if terrain_names is None:
        return torch.zeros_like(terrain_types, dtype=torch.bool)
    wanted = {str(name) for name in names}
    wanted_cols = tuple(i for i, name in enumerate(terrain_names) if str(name) in wanted)
    return _isin(terrain_types, wanted_cols)


def eligible_mpc_reference_envs(
    *,
    num_envs: int,
    device: torch.device,
    terrain_types=None,
    terrain_levels=None,
    terrain_names: list[str] | tuple[str, ...] | None = None,
    cfg: MpcReferenceParticipationCfg,
) -> Tensor:
    """Return envs eligible for MPC reference participation.

    Exclude pairs use AND logic: an env is excluded only when its terrain
    column/name matches the pair and its difficulty row also matches the pair.
    """

    base = torch.ones(int(num_envs), dtype=torch.bool, device=device)
    types = None
    rows = None
    if terrain_types is not None:
        types = _as_1d_long(terrain_types, device=device, num_envs=num_envs, name="terrain_types")

    if terrain_levels is not None:
        rows = _as_1d_long(terrain_levels, device=device, num_envs=num_envs, name="terrain_levels")

    for pair in cfg.exclude_pairs:
        if types is None:
            pair_terrain = torch.ones(int(num_envs), dtype=torch.bool, device=device)
        else:
            terrain_by_col = _isin(types, pair.terrain_cols)
            terrain_by_name = _name_mask(types, terrain_names, pair.terrain_names, none_matches=False)
            pair_terrain = torch.logical_or(terrain_by_col, terrain_by_name)
        if rows is None:
            pair_row = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
        else:
            pair_row = _isin(rows, pair.terrain_rows)
        base &= torch.logical_not(pair_terrain & pair_row)
    return base


def select_mpc_reference_envs(
    *,
    num_envs: int,
    device: torch.device,
    terrain_types=None,
    terrain_levels=None,
    terrain_names: list[str] | tuple[str, ...] | None = None,
    cfg: MpcReferenceParticipationCfg,
    sample_count: int,
    cursor: int = 0,
    return_eligible: bool = False,
):
    eligible = eligible_mpc_reference_envs(
        num_envs=num_envs,
        device=device,
        terrain_types=terrain_types,
        terrain_levels=terrain_levels,
        terrain_names=terrain_names,
        cfg=cfg,
    )
    selected = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
    ids = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
    if not bool(cfg.enabled) or int(ids.numel()) == 0 or int(sample_count) <= 0:
        result = (selected, int(cursor), eligible)
        return result if return_eligible else result[:2]
    if cfg.selection_mode != "round_robin":
        raise ValueError(f"unsupported MPC participation selection_mode: {cfg.selection_mode}")

    count = min(int(sample_count), int(ids.numel()))
    start = int(cursor) % int(ids.numel())
    order = torch.cat((ids[start:], ids[:start]), dim=0)
    selected[order[:count]] = True
    next_cursor = (start + count) % int(ids.numel())
    result = (selected, int(next_cursor), eligible)
    return result if return_eligible else result[:2]


__all__ = [
    "MpcReferenceParticipationCfg",
    "MpcTerrainDifficultyPair",
    "eligible_mpc_reference_envs",
    "select_mpc_reference_envs",
]
