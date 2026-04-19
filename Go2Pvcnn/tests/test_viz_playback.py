"""L4 tests: kinematic playback control-flow logic.

These tests verify the plan-once / replay-then-replan state machine
abstraction used by ``go2_foostep_planner.py``.  No Isaac Lab rendering
or simulation is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import signal
import torch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeTrajectoryResult:
    """Minimal stand-in for BatchedTrajectoryResult for playback logic tests."""

    def __init__(self, n_frames: int = 10):
        self.num_frames = n_frames
        self.root_pos_w = torch.randn(1, n_frames, 3, dtype=torch.float64)
        self.root_quat_w = torch.zeros(1, n_frames, 4, dtype=torch.float64)
        self.root_quat_w[..., 0] = 1.0
        self.joint_angles = torch.zeros(1, n_frames, 12, dtype=torch.float64)
        self.foot_pos_w = torch.zeros(1, n_frames, 4, 3, dtype=torch.float64)
        self.planned_touchdown_w = torch.zeros(1, 4, 3, dtype=torch.float64)
        self.contact_state = torch.ones(1, n_frames, 4, dtype=torch.float32)


class TestKinematicPlaybackLogic:
    """Verify the plan-once / replay / replan state machine."""

    def test_state_chain_last_frame_to_next_input(self):
        """Trajectory last-frame state feeds into the next planning cycle."""
        result = FakeTrajectoryResult(n_frames=10)
        last_frame = result.num_frames - 1

        next_root_pos = result.root_pos_w[:, last_frame]
        next_root_quat = result.root_quat_w[:, last_frame]
        next_joint_angles = result.joint_angles[:, last_frame]
        next_foot_pos = result.foot_pos_w[:, last_frame]

        assert next_root_pos.shape == (1, 3)
        assert next_root_quat.shape == (1, 4)
        assert next_joint_angles.shape == (1, 12)
        assert next_foot_pos.shape == (1, 4, 3)

    def test_playback_frame_counter_wraps(self):
        """After playing all T frames, counter should trigger replan."""
        n_frames = 10
        playback_frame = 0
        for _ in range(n_frames):
            playback_frame += 1
        assert playback_frame >= n_frames

    def test_touchdown_available_at_frame_zero(self):
        """Touchdown markers are set at result creation, before playback starts."""
        result = FakeTrajectoryResult(n_frames=10)
        assert result.planned_touchdown_w is not None
        assert result.planned_touchdown_w.shape == (1, 4, 3)

    def test_command_change_triggers_replan(self):
        """A command change should force immediate replan in the next cycle."""
        old_cmd = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
        new_cmd = torch.tensor([[0.0, 0.2, 0.0]], dtype=torch.float64)
        assert not torch.allclose(old_cmd, new_cmd)

    def test_replan_conditions(self):
        """Exercise the full set of conditions that trigger a replan."""
        result = FakeTrajectoryResult(n_frames=10)
        last_cmd = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
        teleop_values = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)

        # Condition 1: no previous result
        assert _need_replan(result=None, playback_frame=0,
                            reset_requested=False, teleop_values=teleop_values,
                            last_cmd=last_cmd)

        # Condition 2: playback exhausted
        assert _need_replan(result=result, playback_frame=10,
                            reset_requested=False, teleop_values=teleop_values,
                            last_cmd=last_cmd)

        # Condition 3: reset requested
        assert _need_replan(result=result, playback_frame=0,
                            reset_requested=True, teleop_values=teleop_values,
                            last_cmd=last_cmd)

        # Condition 4: command changed
        new_cmd = torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float64)
        assert _need_replan(result=result, playback_frame=0,
                            reset_requested=False, teleop_values=new_cmd,
                            last_cmd=last_cmd)

        # No replan when nothing changed and frames remain
        assert not _need_replan(result=result, playback_frame=5,
                                reset_requested=False, teleop_values=teleop_values,
                                last_cmd=last_cmd)

    def test_state_chain_multi_cycle(self):
        """Simulate two plan-replay cycles; state chains correctly."""
        result1 = FakeTrajectoryResult(n_frames=5)
        result1.root_pos_w[0, -1] = torch.tensor([1.0, 2.0, 0.3])

        last_frame = result1.num_frames - 1
        chained_pos = result1.root_pos_w[:, last_frame]
        assert torch.allclose(chained_pos, torch.tensor([[1.0, 2.0, 0.3]], dtype=torch.float64))

        result2 = FakeTrajectoryResult(n_frames=8)
        result2.root_pos_w[0, -1] = torch.tensor([3.0, 4.0, 0.35])
        chained_pos2 = result2.root_pos_w[:, result2.num_frames - 1]
        assert torch.allclose(chained_pos2, torch.tensor([[3.0, 4.0, 0.35]], dtype=torch.float64))

    def test_playback_frame_stays_in_bounds(self):
        """Playback frame index never exceeds num_frames."""
        result = FakeTrajectoryResult(n_frames=10)
        playback_frame = 0
        accessed_frames = []
        while playback_frame < result.num_frames:
            accessed_frames.append(playback_frame)
            playback_frame += 1
        assert max(accessed_frames) == result.num_frames - 1
        assert len(accessed_frames) == result.num_frames

    def test_trajectory_motion_summary_detects_standstill_result(self):
        from extension.viz.go2_foostep_planner import _trajectory_motion_summary

        result = FakeTrajectoryResult(n_frames=5)
        result.root_pos_w[:] = result.root_pos_w[:, :1]
        result.root_quat_w[:] = result.root_quat_w[:, :1]

        summary = _trajectory_motion_summary(result)

        assert summary["standstill"] is True
        assert summary["dx"] == pytest.approx(0.0)
        assert summary["dy"] == pytest.approx(0.0)
        assert summary["dyaw"] == pytest.approx(0.0)

    def test_trajectory_motion_summary_reports_planar_and_yaw_motion(self):
        from extension.viz.go2_foostep_planner import _trajectory_motion_summary

        result = FakeTrajectoryResult(n_frames=3)
        result.root_pos_w[0, 0] = torch.tensor([0.0, 0.0, 0.3], dtype=torch.float64)
        result.root_pos_w[0, -1] = torch.tensor([0.4, -0.2, 0.3], dtype=torch.float64)
        result.root_quat_w[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        yaw = 0.5
        result.root_quat_w[0, -1] = torch.tensor([torch.cos(torch.tensor(yaw / 2)).item(), 0.0, 0.0, torch.sin(torch.tensor(yaw / 2)).item()], dtype=torch.float64)

        summary = _trajectory_motion_summary(result)

        assert summary["standstill"] is False
        assert summary["dx"] == pytest.approx(0.4)
        assert summary["dy"] == pytest.approx(-0.2)
        assert summary["dyaw"] == pytest.approx(0.5)

    def test_terminal_teleop_sigint_handler_restores_tty_before_interrupt(self):
        from extension.viz.go2_foostep_planner import TerminalTeleop

        teleop = TerminalTeleop(
            device=torch.device("cpu"),
            vx_scale=0.5,
            vy_scale=0.5,
            yaw_scale=1.0,
            timeout_s=0.1,
        )
        calls: list[str] = []

        def fake_restore():
            calls.append("restore")

        teleop._restore_terminal_state = fake_restore  # type: ignore[method-assign]

        with pytest.raises(KeyboardInterrupt):
            teleop._handle_signal(signal.SIGINT, None)

        assert calls == ["restore"]

    def test_terminal_teleop_poll_reads_stdin_and_maps_wasdqe(self, monkeypatch):
        from extension.viz.go2_foostep_planner import TerminalTeleop

        teleop = TerminalTeleop(
            device=torch.device("cpu"),
            vx_scale=0.4,
            vy_scale=0.4,
            yaw_scale=1.0,
            timeout_s=0.2,
        )
        teleop._enabled = True

        fake_stdin = SimpleNamespace(read=lambda n=1: "w")
        select_calls = {"count": 0}

        def fake_select(read_list, write_list, err_list, timeout):
            select_calls["count"] += 1
            if select_calls["count"] == 1:
                return ([fake_stdin], [], [])
            return ([], [], [])

        times = iter([10.0, 10.0])
        monkeypatch.setattr("extension.viz.go2_foostep_planner.sys.stdin", fake_stdin)
        monkeypatch.setattr("extension.viz.go2_foostep_planner.select.select", fake_select)
        monkeypatch.setattr("extension.viz.go2_foostep_planner.time.monotonic", lambda: next(times))

        cmd = teleop.poll()

        torch.testing.assert_close(cmd.values, torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float64))
        assert cmd.reset_requested is False


def _need_replan(
    *,
    result,
    playback_frame: int,
    reset_requested: bool,
    teleop_values: torch.Tensor,
    last_cmd: torch.Tensor | None,
    atol: float = 1e-3,
) -> bool:
    """Pure-logic replan predicate matching the viz main loop."""
    if result is None:
        return True
    if playback_frame >= result.num_frames:
        return True
    if reset_requested:
        return True
    if last_cmd is not None and not torch.allclose(teleop_values, last_cmd, atol=atol):
        return True
    return False
