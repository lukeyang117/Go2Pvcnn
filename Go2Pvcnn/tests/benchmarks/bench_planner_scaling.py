"""L3 scaling benchmarks: burst full replan vs steady-state partial replan.

Reuses synthetic env patterns from ``Go2Pvcnn/scripts/bench_batched_planner.py``.
Timing is collected via ``BatchedTrajectoryManager`` + ``PlannerInstrumentation``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from types import SimpleNamespace
import pytest
import torch

from Go2Pvcnn.scripts.bench_batched_planner import (
    _SyntheticEnv,
    _standstill_env_count,
    _stage_totals,
)
from extension.batched_planner.manager import BatchedTrajectoryManager

# Conservative CUDA-only perf gates (see ``test_perf_thresholds_plan_stage_cuda``).
PERF_THRESHOLDS_MS = {
    1024: {"plan": 100.0},
    1: {"plan": 50.0},
}


def _make_ray_hits(num_envs: int, *, device: torch.device) -> torch.Tensor:
    side = 4
    xs = torch.linspace(-1.0, 1.0, side, dtype=torch.float64, device=device)
    ys = torch.linspace(-1.0, 1.0, side, dtype=torch.float64, device=device)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:  # pragma: no cover
        yy, xx = torch.meshgrid(ys, xs)
    zz = torch.zeros_like(xx)
    grid = torch.stack((xx, yy, zz), dim=-1)
    return grid.unsqueeze(0).expand(num_envs, -1, -1, -1).contiguous()


def _median_stage_totals(per_iter: list[dict[str, float]]) -> dict[str, float]:
    if not per_iter:
        return {}
    keys: set[str] = set()
    for row in per_iter:
        keys.update(row.keys())
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [float(r[k]) for r in per_iter if k in r]
        if vals:
            out[k] = float(statistics.median(vals))
    return out


def _print_median_table(title: str, medians: dict[str, float]) -> None:
    if not medians:
        print(f"{title}: (no stage data)")
        return
    items = sorted(medians.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f"{name}={total_s * 1e3:.3f}ms" for name, total_s in items]
    print(f"{title}: " + " ".join(parts))


@dataclass(frozen=True)
class _BurstResult:
    num_envs: int
    median_stage_s: dict[str, float]
    replanned_envs_last: int


def _run_burst_replan(
    *,
    num_envs: int,
    device: torch.device,
    warmup: int = 2,
    iters: int = 7,
) -> _BurstResult:
    cfg = SimpleNamespace(
        reference_replan_interval_steps=1,
        reference_trajectory_horizon=8,
        dt=0.02,
        planner_instrumentation=True,
        verbose_planner=False,
    )
    manager = BatchedTrajectoryManager(cfg, device=device)
    command = torch.zeros((num_envs, 3), dtype=torch.float64, device=device)
    command[:, 0] = 0.3
    ray_hits = _make_ray_hits(num_envs, device=device)
    episode_length_buf = torch.zeros((num_envs,), dtype=torch.long, device=device)
    env = _SyntheticEnv(episode_length_buf=episode_length_buf, command=command, ray_hits=ray_hits)

    for w in range(max(0, warmup)):
        env.episode_length_buf.fill_(w)
        manager.refresh_from_env(env)
        manager.planner_timing_summary(window=True, reset_window=True)

    per_iter: list[dict[str, float]] = []
    last_replanned = 0
    for i in range(max(1, iters)):
        env.episode_length_buf.fill_(warmup + i)
        replan_mask = manager._compute_replan_mask(env.episode_length_buf, env.command_manager.get_command("base_velocity"))
        last_replanned = int(torch.sum(replan_mask).item())
        manager.refresh_from_env(env)
        summary = manager.planner_timing_summary(window=True, reset_window=True)
        per_iter.append(_stage_totals(summary))

    return _BurstResult(
        num_envs=num_envs,
        median_stage_s=_median_stage_totals(per_iter),
        replanned_envs_last=last_replanned,
    )


@dataclass(frozen=True)
class _SteadyResult:
    n_total: int
    replan_chunk: int
    median_stage_s: dict[str, float]
    last_replan_count: int


def _run_steady_state(
    *,
    device: torch.device,
    n_total: int = 1024,
    replan_interval: int = 10,
    replan_chunk: int | None = None,
    warmup: int = 3,
    iters: int = 5,
) -> _SteadyResult:
    """~10% envs replan per ``refresh`` via rotating ``reset_envs`` + interval cfg.

    ``reference_replan_interval_steps`` is set to ``replan_interval`` so the manager
    matches production steady-state settings; the sparse replan load is enforced by
    resetting a sliding subset of envs each step (same selective subset path as
    sparse interval triggers in integration tests).
    """
    if replan_chunk is None:
        replan_chunk = max(1, n_total // replan_interval)
    cfg = SimpleNamespace(
        reference_replan_interval_steps=int(replan_interval),
        reference_trajectory_horizon=8,
        dt=0.02,
        planner_instrumentation=True,
        verbose_planner=False,
    )
    manager = BatchedTrajectoryManager(cfg, device=device)
    command = torch.zeros((n_total, 3), dtype=torch.float64, device=device)
    command[:, 0] = 0.3
    ray_hits = _make_ray_hits(n_total, device=device)
    episode_length_buf = torch.zeros((n_total,), dtype=torch.long, device=device)
    env = _SyntheticEnv(episode_length_buf=episode_length_buf, command=command, ray_hits=ray_hits)

    # Monotonic global episode counter shared by all envs; keep max step < replan_interval so
    # idle envs never hit interval-based replan (only the rotating reset mask does).
    step = 0
    episode_length_buf.fill_(step)
    manager.refresh_from_env(env)
    manager.planner_timing_summary(window=True, reset_window=True)

    def _rotating_mask(phase: int) -> torch.Tensor:
        mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        start = (phase * replan_chunk) % n_total
        for j in range(replan_chunk):
            mask[(start + j) % n_total] = True
        return mask

    for w in range(max(0, warmup)):
        step += 1
        episode_length_buf.fill_(step)
        manager.reset_envs(_rotating_mask(w))
        manager.refresh_from_env(env)
        manager.planner_timing_summary(window=True, reset_window=True)

    per_iter: list[dict[str, float]] = []
    last_count = 0
    for i in range(max(1, iters)):
        step += 1
        episode_length_buf.fill_(step)
        manager.reset_envs(_rotating_mask(warmup + i))
        replan_mask = manager._compute_replan_mask(episode_length_buf, command)
        last_count = int(torch.sum(replan_mask).item())
        manager.refresh_from_env(env)
        summary = manager.planner_timing_summary(window=True, reset_window=True)
        per_iter.append(_stage_totals(summary))

    return _SteadyResult(
        n_total=n_total,
        replan_chunk=replan_chunk,
        median_stage_s=_median_stage_totals(per_iter),
        last_replan_count=last_count,
    )


def test_burst_replan_scaling_median_stages(bench_device: torch.device) -> None:
    """Burst replan: all envs replan each ``refresh`` for N in {1, 64, 256, 1024}."""
    burst_ns = (1, 64, 256, 1024)
    print(f"\n[bench] burst replan device={bench_device}")
    for n in burst_ns:
        r = _run_burst_replan(num_envs=n, device=bench_device)
        assert r.replanned_envs_last == n, f"expected full replan for N={n}, got {r.replanned_envs_last}"
        _print_median_table(f"burst N={n}", r.median_stage_s)
        assert r.median_stage_s.get("plan", 0.0) > 0.0, "expected non-zero plan stage"


def test_steady_state_partial_replan(bench_device: torch.device) -> None:
    """Steady-state: N=1024, interval=10, ~N/10 envs replan per step via ``reset_envs``."""
    r = _run_steady_state(device=bench_device, n_total=1024, replan_interval=10)
    print(f"\n[bench] steady-state N={r.n_total} chunk={r.replan_chunk} device={bench_device}")
    _print_median_table("steady partial replan", r.median_stage_s)
    assert r.last_replan_count == r.replan_chunk
    assert r.median_stage_s.get("plan", 0.0) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA perf thresholds (optional gate)")
def test_perf_thresholds_plan_stage_cuda() -> None:
    """Separate from collection runs: assert conservative ``plan`` ceilings on CUDA."""
    device = torch.device("cuda:0")
    r1 = _run_burst_replan(num_envs=1, device=device, warmup=2, iters=5)
    plan_ms_1 = r1.median_stage_s.get("plan", 0.0) * 1e3
    assert plan_ms_1 < PERF_THRESHOLDS_MS[1]["plan"], f"N=1 plan median {plan_ms_1:.2f}ms >= threshold"

    r1024 = _run_burst_replan(num_envs=1024, device=device, warmup=2, iters=5)
    plan_ms = r1024.median_stage_s.get("plan", 0.0) * 1e3
    assert plan_ms < PERF_THRESHOLDS_MS[1024]["plan"], f"N=1024 plan median {plan_ms:.2f}ms >= threshold"
