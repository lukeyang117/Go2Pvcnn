"""Device/dtype-local immutable tensors populated before CUDA graph capture."""

from __future__ import annotations

import torch


_CACHE: dict[tuple[str, torch.device, torch.dtype], torch.Tensor] = {}


def constant_like(reference: torch.Tensor, name: str, values) -> torch.Tensor:
    key = (str(name), reference.device, reference.dtype)
    value = _CACHE.get(key)
    if value is None:
        value = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
        _CACHE[key] = value
    return value


__all__ = ["constant_like"]
