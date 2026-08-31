"""Opt-in real Isaac Lab smoke probe for the isolated Parallelism AMP trainer."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path


DEFAULT_CHECKPOINT = Path(
    "/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/"
    "logs/rsl_rl/cross_large_complex_ppo/2026-08-26_17-47-24/11d453a/model_19998.pt"
)
REQUIRED_SCALARS = {
    "AMP/amp_active_ratio",
    "AMP/amp_history_ratio_mean",
    "AMP/amp_value_loss",
    "AMP/discriminator_loss",
    "AMP/approx_kl",
    "AMP/actor_critic_grad_norm",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=str(DEFAULT_CHECKPOINT), help="Pure PPO or full AMP checkpoint")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/parallelism_amp_training_smoke.log"))
    args = parser.parse_args()
    if args.num_envs == 1024 and args.max_iterations == 4 and os.environ.get("RUN_REAL_AMP_1024") != "1":
        print("Refusing 1024x4 Isaac Lab smoke without RUN_REAL_AMP_1024=1", flush=True)
        return 2
    root = Path(__file__).resolve().parents[2]
    launcher = root / "scripts/train_parallelism_amp_cross_large_complex_headless.sh"
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["NUM_ENVS"] = str(args.num_envs)
    env["MAX_ITERATIONS"] = str(args.max_iterations)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        print(f"Checkpoint not found: {checkpoint}", flush=True)
        return 2
    started = time.perf_counter()
    result = subprocess.run(
        ["bash", str(launcher), str(checkpoint)],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - started
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text(result.stdout)
    output = result.stdout
    suspicious_lines = [
        line
        for line in output.splitlines()
        if "observation_space" not in line
        and "action_space" not in line
        and "Box(-inf" not in line
    ]
    bad_output = re.search(
        r"\b(traceback|out of memory|oom|nan|inf(?:inity)?)\b",
        "\n".join(suspicious_lines),
        re.IGNORECASE,
    )
    if result.returncode != 0 or bad_output:
        print(f"AMP smoke failed: returncode={result.returncode}, elapsed={elapsed:.2f}s", flush=True)
        return result.returncode or 1
    iterations = {int(value) for value, _ in re.findall(r"Learning iteration (\d+)/(\d+)", output)}
    expected_iterations = set(range(args.max_iterations))
    if iterations != expected_iterations:
        print(f"AMP smoke iteration mismatch: expected={expected_iterations}, got={iterations}", flush=True)
        return 1

    log_match = re.findall(r"\[Logging\] Directory: (.+)", output)
    if not log_match:
        print("AMP smoke did not report a TensorBoard log directory", flush=True)
        return 1
    run_dir = Path(log_match[-1].strip())
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    event_files = list(run_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"AMP smoke found no TensorBoard event file under {run_dir}", flush=True)
        return 1
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        tags = set()
        scalar_count = 0
        for event_file in event_files:
            accumulator = EventAccumulator(str(event_file))
            accumulator.Reload()
            tags.update(accumulator.Tags().get("scalars", []))
            scalar_count += sum(len(accumulator.Scalars(tag)) for tag in accumulator.Tags().get("scalars", []))
    except Exception as exc:  # pragma: no cover - exercised only by real Isaac smoke
        print(f"Unable to read TensorBoard events: {exc}", flush=True)
        return 1
    missing = sorted(REQUIRED_SCALARS - tags)
    if missing or scalar_count == 0:
        print(f"AMP smoke TensorBoard metrics missing={missing}, scalar_count={scalar_count}", flush=True)
        return 1
    print(
        f"AMP smoke passed: envs={args.num_envs}, iterations={sorted(iterations)}, "
        f"elapsed={elapsed:.2f}s, event_files={len(event_files)}, scalars={scalar_count}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
