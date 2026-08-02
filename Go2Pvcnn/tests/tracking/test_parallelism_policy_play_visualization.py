from __future__ import annotations

import torch
import sys
from types import SimpleNamespace

from scripts import play


def test_panel_command_clamps_to_signed_parallelism_limits() -> None:
    state = play.ParallelismPlayPanelState(vx=3.0, vy=-2.0, vyaw=-4.0)
    command = torch.zeros(1, 3, dtype=torch.float32)

    actual = play._panel_command_tensor(state, command)

    torch.testing.assert_close(actual, torch.tensor([[1.0, -0.5, -1.0]]))


def test_suppressed_termination_stays_diagnostic_but_not_done() -> None:
    raw_masks = {
        "base_contact": torch.tensor([True]),
        "parallelism_ref_joint_pos_too_far": torch.tensor([True]),
    }

    done, diagnostics = play._filter_termination_masks(
        raw_masks,
        {"base_contact": True, "parallelism_ref_joint_pos_too_far": False},
    )

    assert diagnostics["base_contact"].item() is True
    assert diagnostics["parallelism_ref_joint_pos_too_far"].item() is True
    assert done.item() is True


def test_all_suppressed_terminations_produce_no_done() -> None:
    raw_masks = {
        "time_out": torch.tensor([True]),
        "bad_orientation": torch.tensor([True]),
    }

    done, _ = play._filter_termination_masks(raw_masks, {"time_out": True, "bad_orientation": True})

    assert done.tolist() == [False]


def test_termination_checkbox_checked_enables_reset() -> None:
    raw_masks = {"base_contact": torch.tensor([True])}
    state = play.ParallelismPlayPanelState()

    play._set_panel_termination_checkbox(state, "base_contact", True)

    done, _ = play._filter_termination_masks(raw_masks, state.suppress_termination)

    assert done.tolist() == [True]


def test_termination_checkbox_unchecked_suppresses_reset() -> None:
    raw_masks = {"base_contact": torch.tensor([True])}
    state = play.ParallelismPlayPanelState()

    play._set_panel_termination_checkbox(state, "base_contact", False)

    done, _ = play._filter_termination_masks(raw_masks, state.suppress_termination)

    assert done.tolist() == [False]


def test_play_parser_accepts_parallelism_tracking_flat(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["play.py", "--run_dir", "test"])
    parser = play.build_arg_parser()
    experiment = next(action for action in parser._actions if action.dest == "experiment")

    assert "parallelism_tracking_flat" in experiment.choices


def test_panel_command_updates_env0_only() -> None:
    command = torch.zeros(2, 3, dtype=torch.float32)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _name: command))

    play._apply_panel_velocity_command(env, play.ParallelismPlayPanelState(vx=0.4, vy=-0.2, vyaw=0.3))

    torch.testing.assert_close(command, torch.tensor([[0.4, -0.2, 0.3], [0.0, 0.0, 0.0]]))


def test_reference_visual_frame_uses_current_manager_phase() -> None:
    manager = SimpleNamespace(
        horizon=24,
        phase=torch.tensor([3]),
        root_pos_w=torch.arange(24 * 3, dtype=torch.float32).reshape(1, 24, 3),
        root_rpy_w=torch.zeros(1, 24, 3),
        joint_pos=torch.arange(24 * 12, dtype=torch.float32).reshape(1, 24, 12),
        foot_pos_w=torch.arange(24 * 4 * 3, dtype=torch.float32).reshape(1, 24, 4, 3),
        contact_state=torch.ones(1, 24, 4, dtype=torch.bool),
    )

    frame = play._parallelism_visual_frame(manager, env_id=0)

    torch.testing.assert_close(frame.root_pos_w, manager.root_pos_w[0, 3])
    torch.testing.assert_close(frame.joint_pos, manager.joint_pos[0, 3])
    torch.testing.assert_close(frame.foot_pos_w, manager.foot_pos_w[0, 3])
    assert frame.future_foot_pos_w.shape == (21, 4, 3)


def test_termination_filter_suppresses_before_env_reads_reset_buffers() -> None:
    class FakeTerminationManager:
        def __init__(self) -> None:
            self._term_names = ("time_out", "parallelism_ref_joint_pos_too_far")
            self._term_cfgs = (SimpleNamespace(time_out=True), SimpleNamespace(time_out=False))
            self._term_dones = {
                "time_out": torch.zeros(1, dtype=torch.bool),
                "parallelism_ref_joint_pos_too_far": torch.zeros(1, dtype=torch.bool),
            }
            self._truncated_buf = torch.zeros(1, dtype=torch.bool)
            self._terminated_buf = torch.zeros(1, dtype=torch.bool)

        def compute(self):
            self._term_dones["time_out"][:] = True
            self._term_dones["parallelism_ref_joint_pos_too_far"][:] = True
            self._truncated_buf[:] = True
            self._terminated_buf[:] = True
            return self._truncated_buf | self._terminated_buf

    state = play.ParallelismPlayPanelState()
    manager = FakeTerminationManager()
    diagnostics = play._install_parallelism_termination_filter(manager, state)

    done = manager.compute()

    assert diagnostics.raw_masks["time_out"].item() is True
    assert diagnostics.raw_masks["parallelism_ref_joint_pos_too_far"].item() is True
    assert done.item() is False
    assert manager._truncated_buf.item() is False
    assert manager._terminated_buf.item() is False


def test_reference_articulation_receives_current_root_and_joint_frame() -> None:
    class FakeReferenceRobot:
        def __init__(self) -> None:
            self.root_pose = None
            self.root_velocity = None
            self.joint_pos = None
            self.joint_vel = None

        def write_root_pose_to_sim(self, value):
            self.root_pose = value.clone()

        def write_root_velocity_to_sim(self, value):
            self.root_velocity = value.clone()

        def write_joint_state_to_sim(self, joint_pos, joint_vel):
            self.joint_pos = joint_pos.clone()
            self.joint_vel = joint_vel.clone()

    reference_robot = FakeReferenceRobot()
    frame = play.ParallelismVisualFrame(
        root_pos_w=torch.tensor([1.0, 2.0, 0.3]),
        root_rpy_w=torch.zeros(3),
        joint_pos=torch.arange(12, dtype=torch.float32),
        foot_pos_w=torch.zeros(4, 3),
        contact_state=torch.ones(4, dtype=torch.bool),
        future_root_pos_w=torch.zeros(1, 3),
        future_foot_pos_w=torch.zeros(1, 4, 3),
        future_contact_state=torch.ones(1, 4, dtype=torch.bool),
    )

    play._write_parallelism_reference_robot(reference_robot, frame)

    torch.testing.assert_close(reference_robot.root_pose, torch.tensor([[1.0, 2.0, 0.3, 1.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(reference_robot.root_velocity, torch.zeros(1, 6))
    torch.testing.assert_close(reference_robot.joint_pos, torch.arange(12, dtype=torch.float32).unsqueeze(0))
    torch.testing.assert_close(reference_robot.joint_vel, torch.zeros(1, 12))


def test_reference_root_write_uses_current_frame_without_joint_write() -> None:
    class FakeReferenceRobot:
        def __init__(self) -> None:
            self.root_pose = None
            self.root_velocity = None
            self.joint_writes = 0

        def write_root_pose_to_sim(self, value):
            self.root_pose = value.clone()

        def write_root_velocity_to_sim(self, value):
            self.root_velocity = value.clone()

        def write_joint_state_to_sim(self, *_args):
            self.joint_writes += 1

    reference_robot = FakeReferenceRobot()
    frame = play.ParallelismVisualFrame(
        root_pos_w=torch.tensor([1.0, 2.0, 0.3]),
        root_rpy_w=torch.zeros(3),
        joint_pos=torch.zeros(12),
        foot_pos_w=torch.zeros(4, 3),
        contact_state=torch.ones(4, dtype=torch.bool),
        future_root_pos_w=torch.zeros(1, 3),
        future_foot_pos_w=torch.zeros(1, 4, 3),
        future_contact_state=torch.ones(1, 4, dtype=torch.bool),
    )

    play._write_parallelism_reference_root(reference_robot, frame)

    torch.testing.assert_close(reference_robot.root_pose, torch.tensor([[1.0, 2.0, 0.3, 1.0, 0.0, 0.0, 0.0]]))
    assert reference_robot.joint_writes == 0


def test_parallelism_joint_error_data_matches_reference_and_policy_order() -> None:
    robot = SimpleNamespace(
        joint_names=("j0", "j1"),
        data=SimpleNamespace(joint_pos=torch.tensor([[0.4, -0.1]], dtype=torch.float32)),
    )
    env = SimpleNamespace(scene={"robot": robot})
    frame = play.ParallelismVisualFrame(
        root_pos_w=torch.zeros(3),
        root_rpy_w=torch.zeros(3),
        joint_pos=torch.tensor([0.1, 0.2], dtype=torch.float32),
        foot_pos_w=torch.zeros(4, 3),
        contact_state=torch.ones(4, dtype=torch.bool),
        future_root_pos_w=torch.zeros(1, 3),
        future_foot_pos_w=torch.zeros(1, 4, 3),
        future_contact_state=torch.ones(1, 4, dtype=torch.bool),
    )

    names, reference, actual, error = play._parallelism_joint_error_data(env, frame)

    assert names == ("j0", "j1")
    torch.testing.assert_close(reference, torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(actual, torch.tensor([0.4, -0.1]))
    torch.testing.assert_close(error, torch.tensor([0.3, -0.3]))


def test_reference_visualizer_refreshes_manager_before_writing(monkeypatch) -> None:
    visualizer = object.__new__(play._ParallelismPlayVisualizer)

    class FakeManager:
        def __init__(self):
            self.refresh_count = 0

        def refresh(self):
            self.refresh_count += 1

    manager = FakeManager()
    frame = object()
    reference_robot = object()
    base_env = SimpleNamespace(scene={"reference_robot": reference_robot})
    monkeypatch.setattr(play, "_parallelism_visual_frame", lambda _manager: frame)
    written = []
    monkeypatch.setattr(
        play,
        "_write_parallelism_reference_robot",
        lambda robot, value: written.append((robot, value)),
    )

    visualizer.write_reference(base_env, manager)

    assert manager.refresh_count == 1
    assert written == [(reference_robot, frame)]
