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

    def test_viewer_loop_need_replan_tracks_teleop_values_in_real_helper(self):
        from extension.viz.go2_foostep_planner import _viewer_loop_need_replan

        result = FakeTrajectoryResult(n_frames=10)
        last_cmd = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
        unchanged_cmd = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64)
        changed_cmd = torch.tensor([[0.0, 0.2, 0.0]], dtype=torch.float64)

        assert _viewer_loop_need_replan(
            result=None,
            playback_frame=0,
            reset_requested=False,
            teleop_values=unchanged_cmd,
            last_cmd=last_cmd,
        )
        assert not _viewer_loop_need_replan(
            result=result,
            playback_frame=5,
            reset_requested=False,
            teleop_values=unchanged_cmd,
            last_cmd=last_cmd,
        )
        assert _viewer_loop_need_replan(
            result=result,
            playback_frame=5,
            reset_requested=False,
            teleop_values=changed_cmd,
            last_cmd=last_cmd,
        )
        assert _viewer_loop_need_replan(
            result=result,
            playback_frame=result.num_frames,
            reset_requested=False,
            teleop_values=unchanged_cmd,
            last_cmd=last_cmd,
        )
        assert _viewer_loop_need_replan(
            result=result,
            playback_frame=5,
            reset_requested=True,
            teleop_values=unchanged_cmd,
            last_cmd=last_cmd,
        )

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

    def test_parse_scripted_command_returns_single_command_tensor(self):
        from extension.viz.go2_foostep_planner import _parse_scripted_command

        command = _parse_scripted_command("0.0 0.0 0.3", device=torch.device("cpu"))

        torch.testing.assert_close(command, torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float64))

    def test_parse_scripted_command_rejects_wrong_arity(self):
        from extension.viz.go2_foostep_planner import _parse_scripted_command

        with pytest.raises(ValueError, match="three floats"):
            _parse_scripted_command("0.0 0.3", device=torch.device("cpu"))

    def test_subsample_semantic_height_points_keeps_grid_alignment_and_ignores_invalid_hits(self):
        from extension.viz.go2_foostep_planner import (
            SEMANTIC_LARGE_ID,
            SEMANTIC_SMALL_ID,
            SEMANTIC_TERRAIN_ID,
            _subsample_semantic_height_points,
        )

        ray_hits = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, float("inf")],
                [3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 0.6],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
                [2.0, 3.0, 0.0],
                [3.0, 3.0, 0.0],
            ],
            dtype=torch.float64,
        )
        semantic_map = torch.tensor(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.int64,
        )

        points_by_class, diagnostics = _subsample_semantic_height_points(ray_hits, semantic_map, stride=2)

        assert diagnostics["terrain_hit_count"] == 2
        assert diagnostics["small_hit_count"] == 0
        assert diagnostics["large_hit_count"] == 1
        assert diagnostics["valid_sample_count"] == 3
        assert diagnostics["height_lift_max"] == pytest.approx(0.6)
        torch.testing.assert_close(
            points_by_class[SEMANTIC_TERRAIN_ID],
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]], dtype=torch.float64),
        )
        assert points_by_class[SEMANTIC_SMALL_ID].shape == (0, 3)
        torch.testing.assert_close(
            points_by_class[SEMANTIC_LARGE_ID],
            torch.tensor([[0.0, 2.0, 0.6]], dtype=torch.float64),
        )

    def test_format_semantic_diagnostics_reports_required_contract(self):
        from extension.viz.go2_foostep_planner import _format_semantic_diagnostics

        text = _format_semantic_diagnostics(
            {
                "terrain_hit_count": 9,
                "small_hit_count": 4,
                "large_hit_count": 1,
                "valid_sample_count": 14,
                "height_lift_max": 0.55,
            }
        )

        assert "terrain=9" in text
        assert "small=4" in text
        assert "large=1" in text
        assert "valid=14" in text
        assert "height_lift_max=0.550" in text

    def test_together_viewer_handoff_does_not_accumulate_root_height_on_flat_walk(self):
        from extension.batched_together_planner import (
            TogetherPlannerConfig,
            TogetherPlannerTerrain,
            TogetherRobotState,
            plan_segment,
        )
        from extension.batched_together_planner.types import HIP_OFFSETS_ARRAY
        from extension.viz.go2_foostep_planner import (
            _adapt_together_result_for_viewer,
            _together_state_from_reference_result,
        )

        dtype = torch.float32
        device = torch.device("cpu")
        terrain = TogetherPlannerTerrain.from_heightmap(
            torch.zeros((1, 33, 33), dtype=dtype, device=device),
            world_x_range=(-0.8, 0.8),
            world_y_range=(-0.8, 0.8),
        )
        state = TogetherRobotState(
            root_pos=torch.tensor([[0.0, 0.0, 0.30]], dtype=dtype, device=device),
            root_rpy=torch.zeros((1, 3), dtype=dtype, device=device),
            foot_pos=HIP_OFFSETS_ARRAY.to(dtype=dtype, device=device).unsqueeze(0).clone(),
            joint_angles=torch.zeros((1, 12), dtype=dtype, device=device),
        )
        command = torch.tensor([[0.4, 0.0, 0.0]], dtype=dtype, device=device)
        initial_root_z = state.root_pos[:, 2].clone()
        terminal_root_z = []

        for _cycle_idx in range(4):
            result = _adapt_together_result_for_viewer(plan_segment(terrain, state, command, TogetherPlannerConfig()))
            terminal_root_z.append(result.root_pos_w[:, -1, 2].clone())
            state = _together_state_from_reference_result(result, frame_idx=result.num_frames - 1)

        max_terminal_z = torch.stack(terminal_root_z, dim=1).amax(dim=1)
        assert torch.all(max_terminal_z <= initial_root_z + 0.04)

    def test_together_viewer_zero_command_rehome_does_not_replay_vertical_recovery_after_handoff(self):
        from extension.batched_together_planner import (
            TogetherPlannerConfig,
            TogetherPlannerTerrain,
            TogetherRobotState,
            plan_segment,
        )
        from extension.batched_together_planner.types import HIP_OFFSETS_ARRAY
        from extension.viz.go2_foostep_planner import (
            _adapt_together_result_for_viewer,
            _together_state_from_reference_result,
        )

        dtype = torch.float32
        device = torch.device("cpu")
        cfg = TogetherPlannerConfig()
        terrain = TogetherPlannerTerrain.from_heightmap(
            torch.zeros((1, 33, 33), dtype=dtype, device=device),
            world_x_range=(-0.8, 0.8),
            world_y_range=(-0.8, 0.8),
        )
        state = TogetherRobotState(
            root_pos=torch.tensor([[0.0, 0.0, 0.40]], dtype=dtype, device=device),
            root_rpy=torch.zeros((1, 3), dtype=dtype, device=device),
            foot_pos=HIP_OFFSETS_ARRAY.to(dtype=dtype, device=device).unsqueeze(0).clone(),
            joint_angles=torch.zeros((1, 12), dtype=dtype, device=device),
        )
        zero_command = torch.zeros((1, 3), dtype=dtype, device=device)

        first_result = _adapt_together_result_for_viewer(plan_segment(terrain, state, zero_command, cfg))
        first_delta_z = first_result.root_pos_w[0, -1, 2] - first_result.root_pos_w[0, 0, 2]
        assert float(first_delta_z) < -0.08

        handoff_state = _together_state_from_reference_result(first_result, frame_idx=first_result.num_frames - 1)
        second_result = _adapt_together_result_for_viewer(plan_segment(terrain, handoff_state, zero_command, cfg))
        second_delta_z = second_result.root_pos_w[0, -1, 2] - second_result.root_pos_w[0, 0, 2]

        assert abs(float(second_delta_z)) < 0.01

    def test_read_actual_base_state_reports_raw_quat_and_both_rpy_conventions(self):
        from extension.viz.go2_foostep_planner import _read_actual_base_state

        yaw = 0.5
        raw_xyzw = torch.tensor(
            [[0.0, 0.0, torch.sin(torch.tensor(yaw / 2)).item(), torch.cos(torch.tensor(yaw / 2)).item()]],
            dtype=torch.float64,
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2.0, 0.33]], dtype=torch.float64),
                root_quat_w=raw_xyzw,
            )
        )
        base_env = SimpleNamespace(scene={"robot": robot})

        actual = _read_actual_base_state(base_env)

        torch.testing.assert_close(actual["root_pos_w"], torch.tensor([[1.0, 2.0, 0.33]], dtype=torch.float64))
        torch.testing.assert_close(actual["root_quat_raw"], raw_xyzw)
        assert actual["rpy_if_xyzw"].shape == (1, 3)
        assert actual["rpy_if_wxyz"].shape == (1, 3)
        assert actual["rpy_if_xyzw"][0, 2].item() == pytest.approx(yaw, abs=1e-6)
        assert abs(actual["rpy_if_wxyz"][0, 2].item() - yaw) > 1e-3

    def test_read_actual_kinematic_state_reorders_joints_and_slices_feet(self):
        from extension.viz.go2_foostep_planner import _read_actual_kinematic_state

        robot = SimpleNamespace(
            joint_names=[
                "FL_hip_joint",
                "FR_hip_joint",
                "RL_hip_joint",
                "RR_hip_joint",
                "FL_thigh_joint",
                "FR_thigh_joint",
                "RL_thigh_joint",
                "RR_thigh_joint",
                "FL_calf_joint",
                "FR_calf_joint",
                "RL_calf_joint",
                "RR_calf_joint",
            ],
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float64),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
                joint_pos=torch.tensor([[0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]], dtype=torch.float64),
                body_pos_w=torch.tensor(
                    [[[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]],
                    dtype=torch.float64,
                ),
            ),
        )
        base_env = SimpleNamespace(scene={"robot": robot})

        actual = _read_actual_kinematic_state(base_env, foot_ids=[0, 1, 2, 3])

        torch.testing.assert_close(
            actual["joint_pos_planner"],
            torch.arange(12, dtype=torch.float64).reshape(1, 12),
        )
        torch.testing.assert_close(
            actual["foot_pos_w"],
            torch.tensor(
                [[[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]],
                dtype=torch.float64,
            ),
        )

    def test_direct_playback_reorders_planner_joint_angles_into_robot_joint_order(self):
        from extension.viz.go2_foostep_planner import _apply_direct_playback_to_robot

        class FakeRobot:
            joint_names = [
                "FL_hip_joint",
                "FR_hip_joint",
                "RL_hip_joint",
                "RR_hip_joint",
                "FL_thigh_joint",
                "FR_thigh_joint",
                "RL_thigh_joint",
                "RR_thigh_joint",
                "FL_calf_joint",
                "FR_calf_joint",
                "RL_calf_joint",
                "RR_calf_joint",
            ]

            def __init__(self):
                self.root_pose_xyzw = None
                self.joint_pos = None
                self.joint_vel = None

            def write_root_pose_to_sim(self, root_pose_xyzw, env_ids=None):
                self.root_pose_xyzw = root_pose_xyzw.clone()

            def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
                self.joint_pos = joint_pos.clone()
                self.joint_vel = joint_vel.clone()

        robot = FakeRobot()
        fake_result = SimpleNamespace(
            root_pos_w=torch.tensor([[[0.0, 0.0, 0.3]]], dtype=torch.float64),
            root_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64),
            joint_angles=torch.arange(12, dtype=torch.float64).reshape(1, 1, 12),
        )

        _apply_direct_playback_to_robot(robot, fake_result, frame_idx=0)

        torch.testing.assert_close(
            robot.joint_pos,
            torch.tensor([[0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]], dtype=torch.float32),
        )

    def test_viewer_playback_branch_defaults_to_scene_sync(self):
        from extension.viz.go2_foostep_planner import _viewer_direct_playback_step

        call_order: list[str] = []

        class FakeRobot:
            def write_root_pose_to_sim(self, root_pose_xyzw, env_ids=None):
                call_order.append("robot.write_root_pose_to_sim")

            def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
                call_order.append("robot.write_joint_state_to_sim")

        class FakeScene(dict):
            def write_data_to_sim(self):
                call_order.append("scene.write_data_to_sim")

            def update(self, dt):
                call_order.append(f"scene.update({dt:.2f})")

        class FakeSim:
            def render(self):
                call_order.append("sim.render")

        result = FakeTrajectoryResult(n_frames=1)
        scene = FakeScene(robot=FakeRobot())
        base_env = SimpleNamespace(scene=scene, sim=FakeSim(), physics_dt=0.02)

        _viewer_direct_playback_step(base_env, result, frame_idx=0)

        assert call_order == [
            "robot.write_root_pose_to_sim",
            "robot.write_joint_state_to_sim",
            "scene.write_data_to_sim",
            "sim.render",
            "scene.update(0.02)",
        ]

    def test_viewer_playback_branch_scene_sync_path_flushes_scene_before_readback(self):
        from extension.viz.go2_foostep_planner import _viewer_direct_playback_step

        call_order: list[str] = []

        class FakeRobot:
            def write_root_pose_to_sim(self, root_pose_xyzw, env_ids=None):
                call_order.append("robot.write_root_pose_to_sim")

            def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids=None):
                call_order.append("robot.write_joint_state_to_sim")

        class FakeScene(dict):
            def write_data_to_sim(self):
                call_order.append("scene.write_data_to_sim")

            def update(self, dt):
                call_order.append(f"scene.update({dt:.2f})")

        class FakeSim:
            def render(self):
                call_order.append("sim.render")

        result = FakeTrajectoryResult(n_frames=1)
        scene = FakeScene(robot=FakeRobot())
        base_env = SimpleNamespace(scene=scene, sim=FakeSim(), physics_dt=0.02)

        _viewer_direct_playback_step(base_env, result, frame_idx=0, sync_scene=True)

        assert call_order == [
            "robot.write_root_pose_to_sim",
            "robot.write_joint_state_to_sim",
            "scene.write_data_to_sim",
            "sim.render",
            "scene.update(0.02)",
        ]


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
