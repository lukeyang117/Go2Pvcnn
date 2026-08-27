"""Opt-in real Isaac Lab smoke probe for the isolated Parallelism AMP trainer."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Pure PPO or full AMP checkpoint")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/parallelism_amp_training_smoke.log"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    launcher = root / "scripts/train_parallelism_amp_cross_large_complex_headless.sh"
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["NUM_ENVS"] = str(args.num_envs)
    env["MAX_ITERATIONS"] = str(args.max_iterations)
    result = subprocess.run(
        ["bash", str(launcher), str(Path(args.checkpoint).resolve())],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    args.log_file.write_text(result.stdout)
    if result.returncode != 0 or "Traceback" in result.stdout or "OutOfMemory" in result.stdout:
        return result.returncode or 1
    iterations = {int(value) for value, _ in re.findall(r"Learning iteration (\d+)/(\d+)", result.stdout)}
    return 0 if iterations == set(range(args.max_iterations)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
