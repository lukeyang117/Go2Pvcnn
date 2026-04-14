import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.reference.raw_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


def _sample_inputs(n_frames: int = 8):
    torch.manual_seed(42)
    root_pos = torch.randn(n_frames, 3, dtype=torch.float64) * 0.1
    root_pos[:, 2] += 0.35

    q = torch.randn(n_frames, 4, dtype=torch.float64)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)

    foot_targets = torch.randn(n_frames, 4, 3, dtype=torch.float64) * 0.05
    foot_targets[..., 0] += root_pos[:, None, 0]
    foot_targets[..., 1] += root_pos[:, None, 1]
    foot_targets[..., 2] += root_pos[:, None, 2] - 0.25

    return root_pos, q, foot_targets


def _sample_reachable_inputs(n_frames: int = 8):
    from scripts.go2fp.ik import batch_forward_kinematics as raw_batch_forward_kinematics

    root_pos, q, _ = _sample_inputs(n_frames)
    torch.manual_seed(7)
    lower = torch.tensor(
        [-1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227],
        dtype=torch.float64,
    )
    upper = torch.tensor(
        [1.0472, 3.4907, -0.8378, 1.0472, 3.4907, -0.8378, 1.0472, 4.5379, -0.8378, 1.0472, 4.5379, -0.8378],
        dtype=torch.float64,
    )
    joint_angles = lower + torch.rand(n_frames, 12, dtype=torch.float64) * (upper - lower)
    body_pos_w = torch.as_tensor(
        raw_batch_forward_kinematics(
            root_pos.cpu().numpy(),
            q.cpu().numpy(),
            joint_angles.cpu().numpy(),
        ),
        dtype=torch.float64,
    )
    return root_pos, q, body_pos_w[:, 8:, :]


def _sample_roundtrip_inputs():
    root_pos = torch.tensor(
        [
            [0.00, 0.00, 0.35],
            [0.05, -0.02, 0.33],
            [-0.04, 0.03, 0.37],
        ],
        dtype=torch.float64,
    )
    root_quat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    nominal_feet_body = torch.tensor(
        [
            [0.24, 0.14, -0.28],
            [0.24, -0.14, -0.28],
            [-0.24, 0.14, -0.28],
            [-0.24, -0.14, -0.28],
        ],
        dtype=torch.float64,
    )
    frame_offsets = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.01, -0.01, 0.01],
            [-0.01, 0.01, -0.01],
        ],
        dtype=torch.float64,
    )
    foot_targets = nominal_feet_body.unsqueeze(0) + frame_offsets[:, None, :]
    foot_targets = foot_targets + root_pos[:, None, :]
    return root_pos, root_quat, foot_targets


class BatchedIkTest(unittest.TestCase):
    def test_inverse_kinematics_matches_raw(self):
        from extension.batched_planner.ik import batch_inverse_kinematics
        from scripts.go2fp.ik import batch_inverse_kinematics as raw_batch_inverse_kinematics

        root_pos, root_quat, foot_targets = _sample_inputs()

        actual = batch_inverse_kinematics(root_pos, root_quat, foot_targets)
        expected = raw_batch_inverse_kinematics(
            root_pos.cpu().numpy(),
            root_quat.cpu().numpy(),
            foot_targets.cpu().numpy(),
        )

        self.assertEqual(tuple(actual.shape), (root_pos.shape[0], 12))
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), atol=1e-7, rtol=1e-7)

    def test_forward_kinematics_matches_raw(self):
        from extension.batched_planner.ik import batch_forward_kinematics
        from scripts.go2fp.ik import batch_forward_kinematics as raw_batch_forward_kinematics

        root_pos, root_quat, foot_targets = _sample_inputs()
        from scripts.go2fp.ik import batch_inverse_kinematics as raw_batch_inverse_kinematics

        joints = torch.as_tensor(
            raw_batch_inverse_kinematics(
                root_pos.cpu().numpy(),
                root_quat.cpu().numpy(),
                foot_targets.cpu().numpy(),
            ),
            dtype=torch.float64,
        )

        actual = batch_forward_kinematics(root_pos, root_quat, joints)
        expected = raw_batch_forward_kinematics(
            root_pos.cpu().numpy(),
            root_quat.cpu().numpy(),
            joints.cpu().numpy(),
        )

        self.assertEqual(tuple(actual.shape), (root_pos.shape[0], 12, 3))
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), atol=1e-7, rtol=1e-7)

    def test_ik_fk_roundtrip(self):
        from extension.batched_planner.ik import batch_forward_kinematics, batch_inverse_kinematics

        root_pos, root_quat, foot_targets = _sample_roundtrip_inputs()

        joints = batch_inverse_kinematics(root_pos, root_quat, foot_targets)
        body_pos_w = batch_forward_kinematics(root_pos, root_quat, joints)

        self.assertEqual(tuple(body_pos_w.shape), (root_pos.shape[0], 12, 3))
        torch.testing.assert_close(body_pos_w[:, 8:, :], foot_targets, atol=1e-7, rtol=1e-7)

    def test_body_pos_root_relative_matches_raw(self):
        from extension.batched_planner.ik import batch_body_pos_root_relative, batch_forward_kinematics
        from scripts.go2fp.ik import (
            batch_body_pos_root_relative as raw_batch_body_pos_root_relative,
        )
        from scripts.go2fp.ik import batch_inverse_kinematics as raw_batch_inverse_kinematics

        root_pos, root_quat, foot_targets = _sample_inputs()
        joints = torch.as_tensor(
            raw_batch_inverse_kinematics(
                root_pos.cpu().numpy(),
                root_quat.cpu().numpy(),
                foot_targets.cpu().numpy(),
            ),
            dtype=torch.float64,
        )
        body_pos_w = batch_forward_kinematics(root_pos, root_quat, joints)

        actual = batch_body_pos_root_relative(root_pos, root_quat, body_pos_w)
        expected = raw_batch_body_pos_root_relative(
            root_pos.cpu().numpy(),
            root_quat.cpu().numpy(),
            body_pos_w.cpu().numpy(),
        )

        self.assertEqual(tuple(actual.shape), (root_pos.shape[0], 12, 3))
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
