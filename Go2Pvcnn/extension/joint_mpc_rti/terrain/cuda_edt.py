"""Lazy-loaded CUDA exact EDT for fixed 151 x 151 semantic maps."""

from __future__ import annotations

import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load


_GRID_SIZE = 151
_CHANNELS = 2


@lru_cache(maxsize=1)
def _load_extension():
    source_dir = Path(__file__).resolve().parent / "csrc"
    sources = (
        source_dir / "work_efficient_edt.cpp",
        source_dir / "work_efficient_edt_cuda.cu",
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    name = f"joint_mpc_work_efficient_edt_{digest.hexdigest()[:12]}"
    os.environ.setdefault("CUDA_HOME", "/mnt/mydisk/lhy/cuda-12.2")
    python_bin = str(Path(sys.executable).resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join((python_bin, *path_entries))
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    return load(
        name=name,
        sources=[str(source) for source in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def exact_squared_edt_cuda(mask_bcxy: Tensor) -> Tensor:
    """Return exact squared cell distances for bool ``[B,2,151,151]`` CUDA masks."""
    if not isinstance(mask_bcxy, Tensor):
        raise TypeError("mask_bcxy must be a torch.Tensor")
    if not mask_bcxy.is_cuda:
        raise ValueError("mask_bcxy must be a CUDA tensor")
    if mask_bcxy.dtype is not torch.bool:
        raise ValueError("mask_bcxy must have dtype torch.bool")
    if mask_bcxy.ndim != 4 or tuple(mask_bcxy.shape[1:]) != (_CHANNELS, _GRID_SIZE, _GRID_SIZE):
        raise ValueError("mask_bcxy must have shape [B,2,151,151]")
    mask = mask_bcxy.contiguous()
    output = _load_extension().exact_squared_edt(mask)
    if output.dtype is not torch.float32 or output.shape != mask.shape or not output.is_contiguous():
        raise RuntimeError("CUDA exact EDT returned an invalid output contract")
    return output


def semantic_distance_fields_cuda(
    semantic_id: Tensor,
    *,
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    resolution: float,
) -> Tensor:
    """Build exact signed metre distances with output layout ``[2,B,151,151]``."""
    if not isinstance(semantic_id, Tensor):
        raise TypeError("semantic_id must be a torch.Tensor")
    if not semantic_id.is_cuda or semantic_id.dtype is not torch.long:
        raise ValueError("semantic_id must be a CUDA torch.long tensor")
    if semantic_id.ndim != 3 or tuple(semantic_id.shape[1:]) != (_GRID_SIZE, _GRID_SIZE):
        raise ValueError("semantic_id must have shape [B,151,151]")
    if len(small_ids) != 1 or len(large_ids) != 1:
        raise ValueError("CUDA fused semantic EDT currently requires one small id and one large id")
    distance = _load_extension().semantic_distance_fields(
        semantic_id.contiguous(), int(small_ids[0]), int(large_ids[0]), float(resolution)
    )
    return distance


def semantic_distance_fields_out_cuda(
    semantic_id: Tensor,
    distance_out: Tensor,
    vertical_workspace: Tensor,
    *,
    small_ids: tuple[int, ...],
    large_ids: tuple[int, ...],
    resolution: float,
) -> None:
    """Write signed distances into ``[2,B,151,151]`` using ``[4,B,151,151]`` workspace."""
    if len(small_ids) != 1 or len(large_ids) != 1:
        raise ValueError("CUDA fused semantic EDT currently requires one small id and one large id")
    _load_extension().semantic_distance_fields_out(
        semantic_id,
        distance_out,
        vertical_workspace,
        int(small_ids[0]),
        int(large_ids[0]),
        float(resolution),
    )


def copy_height_valid_cuda(
    height_source: Tensor,
    height_out: Tensor,
    valid_out: Tensor,
    origin_source: Tensor,
    origin_out: Tensor,
    yaw_source: Tensor,
    yaw_out: Tensor,
    timestamp_source: Tensor,
    timestamp_out: Tensor,
    version_out: Tensor,
    ready_out: Tensor,
) -> None:
    """Copy height/valid and atomically publish full-batch field metadata."""
    _load_extension().copy_height_valid(
        height_source,
        height_out,
        valid_out,
        origin_source,
        origin_out,
        yaw_source,
        yaw_out,
        timestamp_source,
        timestamp_out,
        version_out,
        ready_out,
    )


__all__ = [
    "exact_squared_edt_cuda",
    "semantic_distance_fields_cuda",
    "semantic_distance_fields_out_cuda",
    "copy_height_valid_cuda",
]
