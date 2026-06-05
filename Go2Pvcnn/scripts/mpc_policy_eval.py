"""Evaluate a trained policy against MPC reference tracking and semantic collisions."""

from __future__ import annotations

import argparse
import json
import sys
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


def command_for_step(args: argparse.Namespace, *, step: int, env_count: int, device: torch.device) -> torch.Tensor:
    del step
    values = [float(v) for v in str(args.command).split()]
    if len(values) != 3:
        raise ValueError("--command must contain exactly three floats: vx vy yaw")
    return torch.tensor(values, dtype=torch.float32, device=device).repeat(int(env_count), 1)


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
