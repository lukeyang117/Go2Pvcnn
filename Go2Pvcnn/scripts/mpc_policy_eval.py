"""Evaluate a trained policy against MPC reference tracking and semantic collisions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch


THIS_FILE = Path(__file__).resolve()
GO2PVCNN_ROOT = THIS_FILE.parent.parent
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for _path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def build_arg_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate policy rollout against MPC reference and semantic collisions.")
    parser.add_argument("--mode", choices=["tracking", "small_collision"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--terrain-rows", type=str, default="0,3,6,9")
    parser.add_argument("--terrain-cols", type=str, default="0")
    parser.add_argument("--command-mode", choices=["fixed", "random", "sweep"], default="fixed")
    parser.add_argument("--command", type=str, default="0.4 0.0 0.0")
    parser.add_argument("--command-sweep", type=str, default="")
    parser.add_argument("--random-command-interval", type=int, default=100)
    parser.add_argument("--small-count-per-tile", type=int, default=80)
    parser.add_argument("--collision-force-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def validate_eval_args(args: argparse.Namespace) -> None:
    if int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args.num_rounds) <= 0:
        raise ValueError("--num-rounds must be positive")
    if int(args.max_steps) < 0:
        raise ValueError("--max-steps must be non-negative")
    if int(args.max_steps) == 0 and int(getattr(args, "livestream", 0)) not in (1, 2):
        raise ValueError("--max-steps 0 is only valid with --livestream 1 or --livestream 2")
    if float(args.collision_force_threshold) < 0.0:
        raise ValueError("--collision-force-threshold must be non-negative")


def parse_command_sweep(value: str) -> list[tuple[float, float, float]]:
    commands: list[tuple[float, float, float]] = []
    for chunk in str(value).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(v) for v in chunk.split()]
        if len(parts) != 3:
            raise ValueError("--command-sweep entries must be 'vx vy yaw'")
        commands.append((parts[0], parts[1], parts[2]))
    if not commands:
        raise ValueError("--command-sweep must contain at least one command in sweep mode")
    return commands


def _command_tuple_from_args(args: argparse.Namespace, *, step: int) -> tuple[float, float, float]:
    mode = str(args.command_mode)
    if mode == "fixed":
        parts = [float(v) for v in str(args.command).split()]
        if len(parts) != 3:
            raise ValueError("--command must contain exactly three floats: vx vy yaw")
        return (parts[0], parts[1], parts[2])
    if mode == "sweep":
        commands = parse_command_sweep(args.command_sweep)
        return commands[int(step) % len(commands)]
    if mode == "random":
        interval = max(1, int(args.random_command_interval))
        bucket = int(step) // interval
        candidates = (
            (0.4, 0.0, 0.0),
            (-0.25, 0.0, 0.0),
            (0.0, 0.3, 0.0),
            (0.0, -0.3, 0.0),
            (0.25, 0.0, 0.5),
            (0.25, 0.0, -0.5),
            (0.2, 0.2, 0.0),
            (0.2, -0.2, 0.0),
        )
        return candidates[bucket % len(candidates)]
    raise ValueError(f"Unsupported command mode: {mode}")


def command_for_step(args: argparse.Namespace, *, step: int, env_count: int, device: torch.device) -> torch.Tensor:
    values = _command_tuple_from_args(args, step=step)
    command = torch.tensor(values, dtype=torch.float64, device=device).repeat(int(env_count), 1)
    return torch.round(command * 1_000_000_000_000) / 1_000_000_000_000


def tracking_foot_metrics(actual_foot_pos_w: torch.Tensor, reference_foot_pos_w: torch.Tensor) -> dict[str, object]:
    actual = torch.as_tensor(actual_foot_pos_w, dtype=torch.float32)
    reference = torch.as_tensor(reference_foot_pos_w, dtype=torch.float32, device=actual.device)
    if actual.shape != reference.shape or actual.ndim != 3 or actual.shape[1:] != (4, 3):
        raise ValueError(f"expected foot tensors with shape [N,4,3], got {tuple(actual.shape)} and {tuple(reference.shape)}")
    error = torch.linalg.norm(actual - reference, dim=-1)
    return {
        "foot_tracking_error_mean_m": float(error.mean().item()),
        "foot_tracking_error_p95_m": float(torch.quantile(error.reshape(-1), 0.95).item()),
        "per_leg_foot_error_mean_m": [float(v) for v in error.mean(dim=0).tolist()],
    }


@dataclass
class SmallCollisionRoundAccumulator:
    num_envs: int
    threshold: float
    device: torch.device
    collided: torch.Tensor = field(init=False)
    first_step: dict[int, int] = field(default_factory=dict)
    body_names_by_env: dict[int, set[str]] = field(default_factory=dict)
    force_max: float = 0.0

    def __post_init__(self) -> None:
        self.collided = torch.zeros((int(self.num_envs),), dtype=torch.bool, device=self.device)

    def update(self, *, step: int, force_matrix_w: torch.Tensor, body_names: tuple[str, ...] | list[str]) -> None:
        force = torch.as_tensor(force_matrix_w, dtype=torch.float32, device=self.device)
        if force.ndim != 4 or force.shape[0] != int(self.num_envs) or force.shape[-1] != 3:
            raise ValueError(f"force_matrix_w must have shape [N,B,F,3], got {tuple(force.shape)}")
        magnitudes = torch.linalg.norm(force, dim=-1)
        active_by_body = magnitudes > float(self.threshold)
        active_env = active_by_body.any(dim=(1, 2))
        self.force_max = max(self.force_max, float(magnitudes.max().item()) if magnitudes.numel() else 0.0)
        active_ids = torch.nonzero(active_env, as_tuple=False).flatten().tolist()
        for env_id in active_ids:
            env_int = int(env_id)
            if env_int not in self.first_step:
                self.first_step[env_int] = int(step)
            self.collided[env_int] = True
            body_ids = torch.nonzero(active_by_body[env_int].any(dim=1), as_tuple=False).flatten().tolist()
            names = self.body_names_by_env.setdefault(env_int, set())
            for body_id in body_ids:
                if int(body_id) < len(body_names):
                    names.add(str(body_names[int(body_id)]))

    def summary(self) -> dict[str, object]:
        count = int(self.collided.sum().item())
        return {
            "collided_env_count": count,
            "num_envs": int(self.num_envs),
            "small_collision_env_rate_per_round": float(count / max(1, int(self.num_envs))),
            "first_collision_step_by_env": {str(k): int(v) for k, v in sorted(self.first_step.items())},
            "collision_body_names_by_env": {str(k): sorted(v) for k, v in sorted(self.body_names_by_env.items())},
            "round_small_force_max": float(self.force_max),
        }


def run_eval(args: argparse.Namespace) -> int:
    validate_eval_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run_eval(args))


if __name__ == "__main__":
    main()
