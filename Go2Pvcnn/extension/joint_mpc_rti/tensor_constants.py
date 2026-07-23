"""Device/dtype-local immutable tensors populated before CUDA graph capture."""

from __future__ import annotations

import torch


_CACHE: dict[tuple[object, ...], torch.Tensor] = {}


def _value_identity(values) -> object:
    if isinstance(values, torch.Tensor):
        return ("tensor", id(values), tuple(values.shape), values.dtype, values.device)
    if isinstance(values, dict):
        return tuple(
            sorted((key, _value_identity(value)) for key, value in values.items())
        )
    if isinstance(values, (list, tuple, range)):
        return tuple(_value_identity(value) for value in values)
    try:
        hash(values)
    except TypeError:
        return repr(values)
    return values


def constant_like(reference: torch.Tensor, name: str, values) -> torch.Tensor:
    key = (str(name), reference.device, reference.dtype, _value_identity(values))
    value = _CACHE.get(key)
    if value is None:
        value = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
        _CACHE[key] = value
    return value


__all__ = ["constant_like"]
