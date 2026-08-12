"""Manual 1024-env training smoke probe for cross-large-complex parallelism tracking."""

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
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    cmd = [
        args.python,
        str(root / "scripts/train.py"),
        "--experiment",
        "parallelism_tracking_cross_large_complex",
        "--num_envs",
        str(args.num_envs),
        "--headless",
        "--max_iterations",
        str(args.max_iterations),
        "--device",
        str(args.device),
    ]
    print("[parallelism_cross_large_complex_training_smoke_probe] Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode != 0:
        return int(result.returncode)
    if "Traceback" in result.stdout or "Learning iteration" not in result.stdout:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
