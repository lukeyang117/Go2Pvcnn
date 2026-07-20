from __future__ import annotations

import sys


def test_watchdog_terminates_only_child_process_group_on_timeout() -> None:
    from .run_monitored_joint_mpc import run_monitored

    result = run_monitored(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_s=0.2,
        heartbeat_seconds=0.05,
    )

    assert result.terminated
    assert result.reason == "hard_timeout"
    assert result.returncode is not None


def test_watchdog_returns_child_output_and_exit_code() -> None:
    from .run_monitored_joint_mpc import run_monitored

    result = run_monitored(
        [sys.executable, "-c", "print('done', flush=True)"],
        timeout_s=5.0,
        heartbeat_seconds=0.05,
    )

    assert not result.terminated
    assert result.returncode == 0
    assert "done" in result.output


def test_watchdog_stops_child_group_on_tree_rss_limit() -> None:
    from .run_monitored_joint_mpc import run_monitored

    result = run_monitored(
        [sys.executable, "-c", "import time; payload=bytearray(8_000_000); time.sleep(60)"],
        timeout_s=5.0,
        heartbeat_seconds=0.02,
        tree_rss_limit_gib=0.001,
    )

    assert result.terminated
    assert result.reason == "tree_rss_limit"
    assert result.snapshots


def test_cpu_only_compiler_growth_requires_no_gpu_progress_for_full_window() -> None:
    from .run_monitored_joint_mpc import ResourceSnapshot, compiler_stalled

    snapshots = (
        ResourceSnapshot(0.0, 10, 1, 100, 0, 0.0, 0.0),
        ResourceSnapshot(31.0, 30, 20, 100, 0, 0.0, 0.0),
    )
    gpu_progress = (
        ResourceSnapshot(0.0, 10, 1, 100, 0, 0.0, 0.0),
        ResourceSnapshot(31.0, 30, 20, 100, 0, 10.0, 5.0),
    )

    assert compiler_stalled(snapshots, window_s=30.0)
    assert not compiler_stalled(gpu_progress, window_s=30.0)
