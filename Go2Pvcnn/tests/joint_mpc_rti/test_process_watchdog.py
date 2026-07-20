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
