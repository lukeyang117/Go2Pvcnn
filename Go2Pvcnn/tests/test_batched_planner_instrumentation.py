import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def advance(self, dt: float) -> None:
        self.t += float(dt)

    def now(self) -> float:
        return self.t


def _fake_result(num_envs: int, num_frames: int):
    from extension.batched_planner.types import BatchedTrajectoryResult

    root_pos = torch.zeros((num_envs, num_frames, 3), dtype=torch.float64)
    root_quat = torch.zeros((num_envs, num_frames, 4), dtype=torch.float64)
    root_quat[..., 0] = 1.0
    joint_angles = torch.zeros((num_envs, num_frames, 12), dtype=torch.float64)
    foot_pos = torch.zeros((num_envs, num_frames, 4, 3), dtype=torch.float64)
    contact_state = torch.ones((num_envs, num_frames, 4), dtype=torch.float32)
    body_pos = torch.zeros((num_envs, num_frames, 12, 3), dtype=torch.float64)
    touchdown = torch.zeros((num_envs, 4, 3), dtype=torch.float64)
    zeros = torch.zeros((num_envs, num_frames, 3), dtype=torch.float64)
    return BatchedTrajectoryResult(
        num_frames=num_frames,
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=zeros.clone(),
        root_ang_vel_w=zeros.clone(),
        joint_angles=joint_angles,
        foot_pos_w=foot_pos.clone(),
        foot_pos_root=foot_pos,
        contact_state=contact_state,
        body_pos_root=body_pos,
        planned_touchdown_w=touchdown,
    )


class _FakeRobot:
    def __init__(self, num_envs: int):
        root_pos = torch.zeros((num_envs, 3), dtype=torch.float64)
        root_quat = torch.zeros((num_envs, 4), dtype=torch.float64)
        root_quat[..., 0] = 1.0
        joint_pos = torch.zeros((num_envs, 12), dtype=torch.float64)
        foot_pos = torch.zeros((num_envs, 4, 3), dtype=torch.float64)
        self.data = SimpleNamespace(
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            joint_pos=joint_pos,
            body_pos_w=foot_pos,
        )

    def find_bodies(self, pattern):
        return torch.tensor([0, 1, 2, 3], dtype=torch.long), ["FL", "FR", "RL", "RR"]


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, name: str):
        return self.command


class _FakeEnv:
    def __init__(self, *, episode_length_buf: torch.Tensor, command: torch.Tensor, ray_hits: torch.Tensor):
        num_envs = int(episode_length_buf.shape[0])
        robot = _FakeRobot(num_envs)
        scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=ray_hits))
        self.scene = SimpleNamespace(robot=robot, sensors={"height_scanner": scanner})
        self.command_manager = _FakeCommandManager(command)
        self.episode_length_buf = episode_length_buf
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.unwrapped = self


class BatchedPlannerInstrumentationTest(unittest.TestCase):
    def test_planner_timing_summary_accumulates_named_stages(self):
        from extension.batched_planner.instrumentation import PlannerInstrumentation

        clock = _FakeClock()
        instr = PlannerInstrumentation(clock=clock.now)

        with instr.stage("terrain"):
            clock.advance(0.010)
        with instr.stage("trajectory"):
            clock.advance(0.025)
        with instr.stage("terrain"):
            clock.advance(0.015)

        summary = instr.summary()
        self.assertIn("terrain", summary.stages)
        self.assertIn("trajectory", summary.stages)
        self.assertEqual(summary.stages["terrain"].count, 2)
        self.assertAlmostEqual(summary.stages["terrain"].total_s, 0.025, places=9)
        self.assertEqual(summary.stages["trajectory"].count, 1)
        self.assertAlmostEqual(summary.stages["trajectory"].total_s, 0.025, places=9)

    def test_quiet_by_default_behavior_is_preserved(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(reference_replan_interval_steps=50, reference_trajectory_horizon=5, dt=0.02)
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        env = _FakeEnv(
            episode_length_buf=torch.tensor([0], dtype=torch.long),
            command=torch.zeros((1, 3), dtype=torch.float64),
            ray_hits=torch.zeros((1, 16, 3), dtype=torch.float64),
        )

        buf = io.StringIO()
        with redirect_stdout(buf), patch(
            "extension.batched_planner.manager.PlannerTerrain.from_ray_hits",
            return_value=SimpleNamespace(),
        ), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            return_value=_fake_result(1, 5),
        ):
            manager.refresh_from_env(env)

        self.assertEqual(buf.getvalue(), "")

    def test_verbose_mode_surfaces_compact_planner_diagnostics(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(
            reference_replan_interval_steps=50,
            reference_trajectory_horizon=5,
            dt=0.02,
            verbose_planner=True,
            verbose_planner_interval_steps=1,
        )
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        env = _FakeEnv(
            episode_length_buf=torch.tensor([0], dtype=torch.long),
            command=torch.zeros((1, 3), dtype=torch.float64),
            ray_hits=torch.zeros((1, 16, 3), dtype=torch.float64),
        )

        buf = io.StringIO()
        with redirect_stdout(buf), patch(
            "extension.batched_planner.manager.PlannerTerrain.from_ray_hits",
            return_value=SimpleNamespace(),
        ), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            return_value=_fake_result(1, 5),
        ):
            manager.refresh_from_env(env)

        out = buf.getvalue()
        self.assertIn("PlannerTiming", out)

    def test_manager_passes_instrumentation_to_planner_entrypoint_when_verbose(self):
        from extension.batched_planner.manager import BatchedTrajectoryManager

        cfg = SimpleNamespace(
            reference_replan_interval_steps=50,
            reference_trajectory_horizon=5,
            dt=0.02,
            verbose_planner=True,
            verbose_planner_interval_steps=1,
        )
        manager = BatchedTrajectoryManager(cfg, device=torch.device("cpu"))
        env = _FakeEnv(
            episode_length_buf=torch.tensor([0], dtype=torch.long),
            command=torch.zeros((1, 3), dtype=torch.float64),
            ray_hits=torch.zeros((1, 16, 3), dtype=torch.float64),
        )

        def fake_generate(*args, **kwargs):
            instr = kwargs.get("instrumentation", None)
            self.assertIsNotNone(instr)
            self.assertTrue(getattr(instr, "enabled", False))
            return _fake_result(1, 5)

        buf = io.StringIO()
        with redirect_stdout(buf), patch(
            "extension.batched_planner.manager.PlannerTerrain.from_ray_hits",
            return_value=SimpleNamespace(),
        ), patch(
            "extension.batched_planner.manager.batched_generate_trajectory",
            side_effect=fake_generate,
        ):
            manager.refresh_from_env(env)


if __name__ == "__main__":
    unittest.main()
