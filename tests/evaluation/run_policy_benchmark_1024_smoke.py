"""Real Isaac acceptance test for one 1024-environment, 48-transition run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-type",
        required=True,
        choices=("amp", "distillation", "ppo", "teacher", "ame", "ame_amp", "semloco"),
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--transitions", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=Path("results/policy_benchmark_smoke"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.num_envs != 1024:
        raise ValueError("the acceptance test must use exactly 1024 environments")
    if not args.checkpoint.is_file():
        raise ValueError("--checkpoint must be an explicit checkpoint file")
    root = Path(__file__).resolve().parents[2]
    command = [str(root / "scripts" / "run_policy_benchmark.sh"), "--experiment-type", args.experiment_type, "--checkpoint", str(args.checkpoint), "--suite", "complex_mixed", "--num-envs", "1024", "--max-steps", str(args.transitions), "--output-dir", str(args.output_dir), "--smoke-test", "--device", args.device, "--seed", str(args.seed), "--headless"]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"benchmark smoke failed:\n{completed.stdout}\n{completed.stderr}")
    output = completed.stdout + completed.stderr
    lowered = output.lower()
    if "traceback" in lowered or "out of memory" in lowered or re.search(r"(?<![a-z])(?:nan|inf)(?![a-z])", lowered):
        raise RuntimeError(f"benchmark smoke emitted numerical/runtime failure:\n{output}")
    summary_paths = list(args.output_dir.glob(f"{args.experiment_type}/complex_mixed/summary.json"))
    if not summary_paths:
        raise RuntimeError("benchmark smoke did not produce summary.json")
    summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    if int(summary.get("valid_windows", 0)) < 1:
        raise RuntimeError("benchmark smoke did not produce a complete valid 24-frame window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
