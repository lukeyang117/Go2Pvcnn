"""Process-group supervisor for heavy joint MPC compile and acceptance commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import signal
import subprocess
import sys
import time


@dataclass(frozen=True)
class MonitoredResult:
    command: tuple[str, ...]
    returncode: int | None
    output: str
    terminated: bool
    reason: str
    elapsed_s: float


def run_monitored(
    command: list[str] | tuple[str, ...],
    *,
    timeout_s: float,
    heartbeat_seconds: float = 5.0,
    terminate_grace_s: float = 1.0,
) -> MonitoredResult:
    argv = tuple(str(value) for value in command)
    started = time.monotonic()
    child = subprocess.Popen(
        argv,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    terminated = False
    reason = "completed"
    output = ""
    try:
        output, _ = child.communicate(timeout=float(timeout_s))
    except subprocess.TimeoutExpired as error:
        terminated = True
        reason = "hard_timeout"
        output = error.output or ""
        os.killpg(child.pid, signal.SIGTERM)
        try:
            tail, _ = child.communicate(timeout=float(terminate_grace_s))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            tail, _ = child.communicate()
        output += tail or ""
    return MonitoredResult(
        command=argv,
        returncode=child.returncode,
        output=output,
        terminated=terminated,
        reason=reason,
        elapsed_s=time.monotonic() - started,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    result = run_monitored(command, timeout_s=args.timeout_seconds, heartbeat_seconds=args.heartbeat_seconds)
    if result.output:
        print(result.output, end="")
    print(f"monitor reason={result.reason} elapsed_s={result.elapsed_s:.3f}", file=sys.stderr)
    return 124 if result.terminated else int(result.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
