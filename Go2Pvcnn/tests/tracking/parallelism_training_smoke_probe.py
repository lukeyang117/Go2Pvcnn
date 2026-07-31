"""Manual 1024-env training smoke probe for parallelism tracking."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cmd = [
        args.python,
        str(root / "scripts/train.py"),
        "--experiment",
        "parallelism_tracking_flat",
        "--num_envs",
        str(args.num_envs),
        "--headless",
        "--max_iterations",
        str(args.max_iterations),
    ]
    print("[parallelism_training_smoke_probe] Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
