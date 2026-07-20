"""Process-group and resource supervisor for heavy joint MPC commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024


@dataclass(frozen=True)
class ResourceSnapshot:
    elapsed_s: float
    tree_rss_bytes: int
    ptxas_rss_bytes: int
    available_bytes: int
    swap_used_bytes: int
    gpu_process_memory_mib: float | None
    gpu_utilization_percent: float | None


@dataclass(frozen=True)
class MonitoredResult:
    command: tuple[str, ...]
    returncode: int | None
    output: str
    terminated: bool
    reason: str
    elapsed_s: float
    snapshots: tuple[ResourceSnapshot, ...]


def compiler_stalled(
    snapshots: tuple[ResourceSnapshot, ...] | list[ResourceSnapshot],
    *,
    window_s: float = 30.0,
) -> bool:
    if len(snapshots) < 2:
        return False
    newest = snapshots[-1]
    if newest.elapsed_s < float(window_s):
        return False
    cutoff = newest.elapsed_s - float(window_s)
    eligible = [snapshot for snapshot in snapshots if snapshot.elapsed_s <= cutoff]
    oldest = eligible[-1] if eligible else snapshots[0]
    compiler_grew = newest.ptxas_rss_bytes > oldest.ptxas_rss_bytes and newest.ptxas_rss_bytes > 0
    gpu_memory_grew = (
        newest.gpu_process_memory_mib is not None
        and oldest.gpu_process_memory_mib is not None
        and newest.gpu_process_memory_mib > oldest.gpu_process_memory_mib
    )
    gpu_active = newest.gpu_utilization_percent is not None and newest.gpu_utilization_percent > 0
    return compiler_grew and not gpu_memory_grew and not gpu_active


def _meminfo() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values.get("MemAvailable", 0), values.get("SwapTotal", 0) - values.get("SwapFree", 0)


def _process_tree(root_pid: int) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text().splitlines()
            fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in status if ":" in line}
            rows.append((int(entry.name), int(fields.get("PPid", "0")), fields.get("Name", "")))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid, _ in rows:
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return [row for row in rows if row[0] in descendants]


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return 0


def _gpu_snapshot(pids: set[int], gpu_index: int | None) -> tuple[float | None, float | None]:
    if gpu_index is None:
        return None, None
    try:
        process_output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout
        memory = 0.0
        for line in process_output.splitlines():
            columns = [column.strip() for column in line.split(",")]
            if len(columns) >= 2 and columns[0].isdigit() and int(columns[0]) in pids:
                memory += float(columns[1])
        utilization_output = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout.strip()
        return memory, float(utilization_output.splitlines()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


def _snapshot(root_pid: int, started: float, gpu_index: int | None) -> ResourceSnapshot:
    tree = _process_tree(root_pid)
    rss = {pid: _rss_bytes(pid) for pid, _, _ in tree}
    available, swap_used = _meminfo()
    gpu_memory, gpu_utilization = _gpu_snapshot(set(rss), gpu_index)
    names = {pid: name for pid, _, name in tree}
    return ResourceSnapshot(
        elapsed_s=time.monotonic() - started,
        tree_rss_bytes=sum(rss.values()),
        ptxas_rss_bytes=sum(value for pid, value in rss.items() if names.get(pid) == "ptxas"),
        available_bytes=available,
        swap_used_bytes=swap_used,
        gpu_process_memory_mib=gpu_memory,
        gpu_utilization_percent=gpu_utilization,
    )


def _terminate_group(child: subprocess.Popen[str], grace_s: float) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def run_monitored(
    command: list[str] | tuple[str, ...],
    *,
    timeout_s: float,
    heartbeat_seconds: float = 5.0,
    terminate_grace_s: float = 1.0,
    tree_rss_limit_gib: float = 16.0,
    ptxas_rss_limit_gib: float = 8.0,
    available_drop_limit_gib: float = 16.0,
    swap_growth_limit_mib: float = 256.0,
    compiler_growth_timeout_s: float = 30.0,
    gpu_index: int | None = None,
) -> MonitoredResult:
    argv = tuple(str(value) for value in command)
    started = time.monotonic()
    baseline_available, baseline_swap = _meminfo()
    child = subprocess.Popen(
        argv,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert child.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    output: list[str] = []
    snapshots: list[ResourceSnapshot] = []
    terminated = False
    reason = "completed"
    next_heartbeat = started
    while child.poll() is None:
        now = time.monotonic()
        for key, _ in selector.select(timeout=min(0.05, max(0.0, next_heartbeat - now))):
            line = key.fileobj.readline()
            if line:
                output.append(line)
        elapsed = now - started
        if elapsed >= float(timeout_s):
            terminated, reason = True, "hard_timeout"
            break
        if now >= next_heartbeat:
            snapshot = _snapshot(child.pid, started, gpu_index)
            snapshots.append(snapshot)
            print(
                "heartbeat "
                f"wall={snapshot.elapsed_s:.1f}s tree_rss={snapshot.tree_rss_bytes / MIB:.1f}MiB "
                f"ptxas_rss={snapshot.ptxas_rss_bytes / MIB:.1f}MiB "
                f"gpu_mem={snapshot.gpu_process_memory_mib}MiB gpu_util={snapshot.gpu_utilization_percent}%",
                file=sys.stderr,
                flush=True,
            )
            if snapshot.tree_rss_bytes > tree_rss_limit_gib * GIB:
                terminated, reason = True, "tree_rss_limit"
            elif snapshot.ptxas_rss_bytes > ptxas_rss_limit_gib * GIB:
                terminated, reason = True, "ptxas_rss_limit"
            elif baseline_available - snapshot.available_bytes > available_drop_limit_gib * GIB:
                terminated, reason = True, "available_memory_drop"
            elif snapshot.swap_used_bytes - baseline_swap > swap_growth_limit_mib * MIB:
                terminated, reason = True, "swap_growth"
            elif compiler_stalled(snapshots, window_s=compiler_growth_timeout_s):
                terminated, reason = True, "compiler_growth_without_gpu_progress"
            if terminated:
                break
            next_heartbeat = now + float(heartbeat_seconds)
    if terminated:
        _terminate_group(child, float(terminate_grace_s))
    tail = child.stdout.read()
    if tail:
        output.append(tail)
    selector.close()
    return MonitoredResult(
        command=argv,
        returncode=child.returncode,
        output="".join(output),
        terminated=terminated,
        reason=reason,
        elapsed_s=time.monotonic() - started,
        snapshots=tuple(snapshots),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--report-json")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a child command is required after --")
    result = run_monitored(
        command,
        timeout_s=args.timeout_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        gpu_index=args.gpu_index,
    )
    if result.output:
        print(result.output, end="")
    payload = {**asdict(result), "snapshots": [asdict(value) for value in result.snapshots]}
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"monitor reason={result.reason} elapsed_s={result.elapsed_s:.3f}", file=sys.stderr)
    return 124 if result.terminated else int(result.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
