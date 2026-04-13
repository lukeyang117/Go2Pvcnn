import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.planner.runtime.raw_go2fp_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


class FlatTerrain:
    def height_at(self, x, y=None):
        if y is None:
            points = torch.as_tensor(x, dtype=torch.float64)
            return torch.zeros(points.shape[:-1], dtype=torch.float64, device=points.device)
        return 0.0

    def roughness_at(self, x, y=None):
        if y is None:
            points = torch.as_tensor(x, dtype=torch.float64)
            return torch.zeros(points.shape[:-1], dtype=torch.float64, device=points.device)
        return 0.0

    def max_height_along_segment(self, p0, p1=None):
        if p1 is None:
            p0_t = torch.as_tensor(p0, dtype=torch.float64)
            return torch.zeros(p0_t.shape[0], dtype=torch.float64, device=p0_t.device)
        if not isinstance(p0, tuple):
            p0_t = torch.as_tensor(p0, dtype=torch.float64)
            return torch.zeros(p0_t.shape[0], dtype=torch.float64, device=p0_t.device)
        return 0.0


class BatchedTrajectoryBatchTest(unittest.TestCase):
    def test_batch_consistency_against_single_env_runs(self):
        from extension.batched_planner.config import BatchedTrajectoryConfig
        from extension.batched_planner.trajectory import batched_generate_trajectory
        from extension.batched_planner.types import BatchedRobotState
        from scripts.go2fp.trajectory import default_initial_state

        terrain = FlatTerrain()
        cfg = BatchedTrajectoryConfig(step_freq=2.0, duty_factor=0.6)

        states = []
        commands = []
        for idx in range(6):
            raw_state = default_initial_state(None, x=0.1 * idx, y=-0.05 * idx)
            states.append(raw_state)
            commands.append(
                [
                    [0.35, 0.0, 0.1],
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                    [0.25, 0.03, -0.08],
                    [0.4, -0.02, 0.12],
                    [0.2, 0.0, 0.0],
                ][idx]
            )

        batched_state = BatchedRobotState(
            root_pos=torch.stack([torch.as_tensor(s.root_pos, dtype=torch.float64) for s in states], dim=0),
            root_quat=torch.stack([torch.as_tensor(s.root_quat, dtype=torch.float64) for s in states], dim=0),
            joint_angles=torch.stack([torch.as_tensor(s.joint_angles, dtype=torch.float64) for s in states], dim=0),
            foot_pos=torch.stack([torch.as_tensor(s.foot_pos, dtype=torch.float64) for s in states], dim=0),
        )
        batched_commands = torch.tensor(commands, dtype=torch.float64)

        batched = batched_generate_trajectory(terrain, batched_state, batched_commands, requested_n_frames=20, dt=0.02, cfg=cfg)

        for idx in range(len(states)):
            single_state = BatchedRobotState(
                root_pos=batched_state.root_pos[idx : idx + 1],
                root_quat=batched_state.root_quat[idx : idx + 1],
                joint_angles=batched_state.joint_angles[idx : idx + 1],
                foot_pos=batched_state.foot_pos[idx : idx + 1],
            )
            single = batched_generate_trajectory(
                terrain,
                single_state,
                batched_commands[idx : idx + 1],
                requested_n_frames=20,
                dt=0.02,
                cfg=cfg,
            )

            self.assertEqual(batched.num_frames, single.num_frames)
            torch.testing.assert_close(batched.root_pos_w[idx], single.root_pos_w[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(batched.root_quat_w[idx], single.root_quat_w[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(batched.joint_angles[idx], single.joint_angles[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(batched.contact_state[idx], single.contact_state[0])
            torch.testing.assert_close(batched.planned_touchdown_w[idx], single.planned_touchdown_w[0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
