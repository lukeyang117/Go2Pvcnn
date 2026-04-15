#!/usr/bin/env python3
"""Offline benchmark entrypoint for the batched planner.

This script is intentionally import-safe:
- CLI/parser helpers have no Isaac Lab dependencies
- Benchmark runtime only imports planner modules when `main()` executes

The benchmark emits JSONL rows containing (at minimum):
- absolute time (seconds) per iteration
- per-env time (seconds)
- standstill env count (derived from command batch)
- replanned env count (derived from manager replan mask for the iteration)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Iterable

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
REPO_ROOT = GO2PVCNN_ROOT.parent
# Keep the Go2Pvcnn module root available for `extension.*` imports, and also
# add the repo root so we can import our own scripts as `Go2Pvcnn.scripts.*`
# even if a raw `scripts` package is present on `sys.path`.
for p in (str(REPO_ROOT), str(GO2PVCNN_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


DEFAULT_ENV_COUNTS: list[int] = [1, 16, 64, 100, 256, 512, 1024, 2048]
DEFAULT_OUTPUT_ROOT = os.path.join("logs", "planner", "batched_planner")
DEFAULT_RESULTS_FILENAME = "planner_bench.jsonl"


def _parse_int_list_csv(value: str) -> list[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return []
    out: list[int] = []
    for p in parts:
        out.append(int(p))
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark batched planner with env-count sweep.")
    parser.add_argument(
        "--env-counts",
        dest="env_counts",
        type=_parse_int_list_csv,
        default=list(DEFAULT_ENV_COUNTS),
        help="Comma-separated env counts to sweep (default matches required sweep).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per env count.")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations per env count.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for tensors (e.g. cpu, cuda:0).")
    parser.add_argument(
        "--output_root",
        "--output-root",
        dest="output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory to write benchmark runs under.",
    )
    parser.add_argument(
        "--run_name",
        "--run-name",
        dest="run_name",
        type=str,
        default=None,
        help="Optional run directory name (defaults to timestamp).",
    )
    parser.add_argument(
        "--results_filename",
        "--results-filename",
        dest="results_filename",
        type=str,
        default=DEFAULT_RESULTS_FILENAME,
        help="JSONL filename written inside the run directory.",
    )
    return parser


def resolve_output_dir(args: argparse.Namespace, *, now: datetime | None = None, mkdir: bool = True) -> Path:
    # Reuse the same run-directory layout used by training so tooling can rely on it.
    from Go2Pvcnn.scripts import train

    return Path(
        train.build_run_log_dir(
            log_root_path=str(getattr(args, "output_root")),
            run_name=getattr(args, "run_name", None),
            now=now,
            mkdir=mkdir,
        )
    )


def results_path(output_dir: Path, *, filename: str = DEFAULT_RESULTS_FILENAME) -> Path:
    return Path(output_dir) / str(filename)


@dataclass(frozen=True)
class _BenchRow:
    num_envs: int
    iter_idx: int
    total_s: float
    per_env_s: float
    standstill_envs: int
    replanned_envs: int
    stages_total_s: dict[str, float]

    def to_json(self) -> dict:
        return {
            "num_envs": int(self.num_envs),
            "iter_idx": int(self.iter_idx),
            "total_s": float(self.total_s),
            "per_env_s": float(self.per_env_s),
            "standstill_envs": int(self.standstill_envs),
            "replanned_envs": int(self.replanned_envs),
            "stages_total_s": {str(k): float(v) for k, v in self.stages_total_s.items()},
        }


class _SyntheticRobot:
    def __init__(self, num_envs: int, *, device: torch.device):
        root_pos = torch.zeros((num_envs, 3), dtype=torch.float64, device=device)
        root_quat = torch.zeros((num_envs, 4), dtype=torch.float64, device=device)
        root_quat[..., 0] = 1.0
        joint_pos = torch.zeros((num_envs, 12), dtype=torch.float64, device=device)
        body_pos = torch.zeros((num_envs, 4, 3), dtype=torch.float64, device=device)
        self.data = SimpleNamespace(
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            joint_pos=joint_pos,
            body_pos_w=body_pos,
        )

    def find_bodies(self, pattern):
        # Compatible with the manager's `robot.find_bodies(".*_foot")` usage.
        return torch.tensor([0, 1, 2, 3], dtype=torch.long, device=self.data.root_pos_w.device), ["FL", "FR", "RL", "RR"]


class _SyntheticCommandManager:
    def __init__(self, command: torch.Tensor):
        self._command = command

    def get_command(self, name: str):
        return self._command


class _SyntheticEnv:
    def __init__(self, *, episode_length_buf: torch.Tensor, command: torch.Tensor, ray_hits: torch.Tensor):
        num_envs = int(episode_length_buf.shape[0])
        device = episode_length_buf.device
        robot = _SyntheticRobot(num_envs, device=device)
        scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=ray_hits))
        self.scene = SimpleNamespace(robot=robot, sensors={"height_scanner": scanner})
        self.command_manager = _SyntheticCommandManager(command)
        self.episode_length_buf = episode_length_buf
        self.device = device
        self.num_envs = num_envs
        self.unwrapped = self


def _standstill_env_count(commands: torch.Tensor, *, eps: float = 1e-5) -> int:
    commands = torch.as_tensor(commands)
    if commands.numel() == 0:
        return 0
    mask = torch.all(torch.abs(commands) <= float(eps), dim=-1)
    return int(torch.sum(mask).item())


def _stage_totals(summary) -> dict[str, float]:
    stages = getattr(summary, "stages", {}) or {}
    out: dict[str, float] = {}
    for name, st in stages.items():
        total = getattr(st, "total_s", None)
        if total is None:
            continue
        out[str(name)] = float(total)
    return out


def _run_env_count(*, num_envs: int, iters: int, warmup: int, device: torch.device) -> Iterable[_BenchRow]:
    from extension.batched_planner.manager import BatchedTrajectoryManager

    cfg = SimpleNamespace(
        reference_replan_interval_steps=1,  # force replan each iteration via episode_length_buf advancement
        reference_trajectory_horizon=8,
        dt=0.02,
        planner_instrumentation=True,  # enable instrumentation without printing
        verbose_planner=False,
    )
    manager = BatchedTrajectoryManager(cfg, device=device)

    # Non-standstill command to avoid fast-path always-standstill trajectories.
    command = torch.zeros((num_envs, 3), dtype=torch.float64, device=device)
    command[:, 0] = 0.3

    # `PlannerTerrain.from_ray_hits(...)` derives world ranges from hit x/y coordinates.
    # Ensure we generate a non-degenerate grid so ranges are strictly increasing.
    side = 4  # 4x4 => 16 rays (matches the manager test configuration)
    xs = torch.linspace(-1.0, 1.0, side, dtype=torch.float64, device=device)
    ys = torch.linspace(-1.0, 1.0, side, dtype=torch.float64, device=device)
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:  # pragma: no cover - compatibility with older torch
        yy, xx = torch.meshgrid(ys, xs)
    zz = torch.zeros_like(xx)
    grid = torch.stack((xx, yy, zz), dim=-1)  # (H, W, 3)
    ray_hits = grid.unsqueeze(0).expand(num_envs, -1, -1, -1).contiguous()

    episode_length_buf = torch.zeros((num_envs,), dtype=torch.long, device=device)
    env = _SyntheticEnv(episode_length_buf=episode_length_buf, command=command, ray_hits=ray_hits)

    # Warmup
    for w in range(max(0, int(warmup))):
        env.episode_length_buf = torch.full((num_envs,), w, dtype=torch.long, device=device)
        manager.refresh_from_env(env)

    for i in range(max(1, int(iters))):
        env.episode_length_buf = torch.full((num_envs,), i + warmup, dtype=torch.long, device=device)

        # Approximate replan count with the same mask logic the manager uses.
        replan_mask = manager._compute_replan_mask(env.episode_length_buf, env.command_manager.get_command("base_velocity"))
        replanned_envs = int(torch.sum(replan_mask).item())

        t0 = perf_counter()
        manager.refresh_from_env(env)
        dt_s = perf_counter() - t0

        summary = manager.planner_timing_summary(window=True, reset_window=True)
        yield _BenchRow(
            num_envs=num_envs,
            iter_idx=i,
            total_s=float(dt_s),
            per_env_s=float(dt_s) / float(max(1, num_envs)),
            standstill_envs=_standstill_env_count(command),
            replanned_envs=replanned_envs,
            stages_total_s=_stage_totals(summary),
        )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(str(getattr(args, "device")))

    out_dir = resolve_output_dir(args, mkdir=True)
    out_file = results_path(out_dir, filename=str(getattr(args, "results_filename")))
    out_file.parent.mkdir(parents=True, exist_ok=True)

    env_counts = list(getattr(args, "env_counts") or [])
    if not env_counts:
        raise SystemExit("--env-counts must contain at least one value")

    rows: list[_BenchRow] = []
    for n in env_counts:
        for row in _run_env_count(
            num_envs=int(n),
            iters=int(getattr(args, "iters")),
            warmup=int(getattr(args, "warmup")),
            device=device,
        ):
            rows.append(row)

    with open(out_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), sort_keys=True) + "\n")

    print(f"[bench_batched_planner] Wrote {len(rows)} rows to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
