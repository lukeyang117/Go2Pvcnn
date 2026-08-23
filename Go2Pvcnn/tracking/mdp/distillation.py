"""GPU-parallel context terms for terrain-aware teacher distillation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager


def terrain_imitation_context_from_metadata(
    terrain_types: torch.Tensor,
    terrain_levels: torch.Tensor,
    terrain_column_names: Sequence[str],
    end_multipliers: Mapping[str, float],
    powers: Mapping[str, float],
    num_rows: int,
    plan_valid: torch.Tensor,
) -> torch.Tensor:
    """Compute [terrain multiplier, plan-valid] without environment-side loops."""

    types = torch.as_tensor(terrain_types, dtype=torch.long)
    if types.ndim != 1:
        types = types.reshape(-1)
    levels = torch.as_tensor(terrain_levels, dtype=torch.float32, device=types.device).reshape(-1)
    valid = torch.as_tensor(plan_valid, dtype=torch.float32, device=types.device).reshape(-1)
    if not (types.numel() == levels.numel() == valid.numel()):
        raise ValueError("terrain_types, terrain_levels, and plan_valid must have the same number of environments")

    difficulty = (levels / max(int(num_rows) - 1, 1)).clamp(0.0, 1.0)
    multiplier = torch.zeros_like(difficulty)
    known = torch.zeros_like(difficulty, dtype=torch.bool)
    for column, name in enumerate(terrain_column_names):
        name = str(name)
        if name not in end_multipliers:
            continue
        end = float(end_multipliers[name])
        power = max(float(powers.get(name, 1.0)), 1.0e-6)
        value = end + (1.0 - end) * torch.pow(1.0 - difficulty, power)
        matches = types == column
        known = torch.logical_or(known, matches)
        multiplier = torch.where(matches, value, multiplier)

    valid = valid.clamp(0.0, 1.0) * known.to(dtype=valid.dtype)
    return torch.stack((multiplier * valid, valid), dim=-1)


def _terrain_generator(env):
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator is None:
        scene_cfg = getattr(getattr(getattr(env, "cfg", None), "scene", None), "terrain", None)
        generator = getattr(scene_cfg, "terrain_generator", None)
    return terrain, generator


def _generated_column_names(generator) -> tuple[str, ...] | None:
    sub_terrains = getattr(generator, "sub_terrains", None)
    num_cols = int(getattr(generator, "num_cols", 0) or 0)
    if not isinstance(sub_terrains, dict) or num_cols <= 0:
        return None

    weighted_names: list[tuple[str, float]] = []
    total = 0.0
    for name, cfg in sub_terrains.items():
        proportion = float(getattr(cfg, "proportion", 0.0) or 0.0)
        if proportion > 0.0:
            weighted_names.append((str(name), proportion))
            total += proportion
    if not weighted_names or total <= 0.0:
        return None

    columns = [weighted_names[-1][0]] * num_cols
    cumulative = 0.0
    for index, (name, proportion) in enumerate(weighted_names):
        start = int(round(num_cols * cumulative / total))
        cumulative += proportion
        end = num_cols if index == len(weighted_names) - 1 else int(round(num_cols * cumulative / total))
        for column in range(max(start, 0), min(end, num_cols)):
            columns[column] = name
    return tuple(columns)


def parallelism_distillation_context(
    env,
    end_multipliers: Mapping[str, float],
    powers: Mapping[str, float],
    num_rows: int = 10,
) -> torch.Tensor:
    """Return the per-environment terrain multiplier and current plan validity."""

    device = torch.device(getattr(env, "device", "cpu"))
    terrain, generator = _terrain_generator(env)
    column_names = _generated_column_names(generator) if generator is not None else None
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if column_names is None or terrain_types is None or terrain_levels is None:
        return torch.zeros((int(getattr(env, "num_envs", 0)), 2), dtype=torch.float32, device=device)

    manager = get_parallelism_reference_manager(env)
    plan_valid = getattr(manager, "step_plan_valid", getattr(manager, "plan_valid", None))
    if plan_valid is None:
        return torch.zeros((int(getattr(env, "num_envs", 0)), 2), dtype=torch.float32, device=device)

    return terrain_imitation_context_from_metadata(
        torch.as_tensor(terrain_types, dtype=torch.long, device=device),
        torch.as_tensor(terrain_levels, dtype=torch.float32, device=device),
        column_names,
        end_multipliers,
        powers,
        num_rows,
        torch.as_tensor(plan_valid, dtype=torch.float32, device=device),
    )
