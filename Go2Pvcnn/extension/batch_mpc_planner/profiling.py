"""Lightweight timing helpers for MPC diagnostics."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import torch


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def should_profile_mpc(cfg) -> bool:
    diagnostics = getattr(cfg, "diagnostics", None)
    if diagnostics is None or not bool(getattr(diagnostics, "emit_runtime_counters", False)):
        return False
    return _env_int("T302G_MPC_PROFILE_LIMIT", 5) > 0


@dataclass
class MpcProfile:
    sync_cuda: bool = False
    stages_ms: dict[str, float] = field(default_factory=dict)
    loss_ms: dict[str, float] = field(default_factory=dict)
    optimize_iters: int = 0
    batch_size: int = 0
    horizon: int = 0

    def now(self) -> float:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def add_stage(self, name: str, elapsed_ms: float) -> None:
        self.stages_ms[name] = self.stages_ms.get(name, 0.0) + float(elapsed_ms)

    def add_loss(self, name: str, elapsed_ms: float) -> None:
        self.loss_ms[name] = self.loss_ms.get(name, 0.0) + float(elapsed_ms)

    def time_block(self, name: str, fn: Callable):
        start = self.now()
        out = fn()
        self.add_stage(name, (self.now() - start) * 1000.0)
        return out


_printed_profiles = 0


def maybe_print_mpc_profile(profile: MpcProfile, *, cfg) -> None:
    global _printed_profiles
    limit = _env_int("T302G_MPC_PROFILE_LIMIT", 5)
    if _printed_profiles >= limit:
        return
    _printed_profiles += 1
    term_limit = max(0, _env_int("T302G_MPC_PROFILE_LOSS_TERMS", 12))
    stage_parts = [f"{name}_ms={value:.3f}" for name, value in sorted(profile.stages_ms.items())]
    loss_total = profile.loss_ms.get("total", 0.0)
    loss_parts = [
        f"{name}_ms={value:.3f}"
        for name, value in sorted(profile.loss_ms.items(), key=lambda item: item[1], reverse=True)
        if name != "total"
    ][:term_limit]
    print(
        "[MPC profile] "
        f"#{_printed_profiles}/{limit} "
        f"profile={getattr(cfg, 'profile_name', 'unknown')} "
        f"batch={profile.batch_size} horizon={profile.horizon} iters={profile.optimize_iters} "
        + " ".join(stage_parts)
        + f" loss.total_ms={loss_total:.3f}"
        + (" loss.top_terms=" + ",".join(loss_parts) if loss_parts else ""),
        flush=True,
    )


__all__ = ["MpcProfile", "maybe_print_mpc_profile", "should_profile_mpc"]
