import unittest

import torch


class ExtensionMdpUtilsTest(unittest.TestCase):
    def test_downsample_height_map_uses_area_pooling(self):
        from Go2Pvcnn.extension.mdp.observations import downsample_height_map

        height_map = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        )

        result = downsample_height_map(height_map, target_size=2)

        expected = torch.tensor([[[3.5, 5.5], [11.5, 13.5]]])
        self.assertEqual(result.shape, (1, 2, 2))
        self.assertTrue(torch.allclose(result, expected))

    def test_compute_tracking_metrics_returns_expected_scalars(self):
        from Go2Pvcnn.extension.mdp.metrics import compute_tracking_metrics

        metrics = compute_tracking_metrics(
            root_pos=torch.tensor([[1.0, 2.0, 0.5]]),
            ref_root_pos=torch.tensor([[2.0, 4.0, 1.0]]),
            root_yaw=torch.tensor([0.1]),
            ref_root_yaw=torch.tensor([0.3]),
            joint_pos=torch.tensor([[0.0, 1.0]]),
            ref_joint_pos=torch.tensor([[1.0, 3.0]]),
            foot_pos_root=torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            ref_foot_pos_root=torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
            contact_state=torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
            ref_contact_state=torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
            touchdown_pos=torch.tensor([[[0.0, 0.0, 0.0]]]),
            ref_touchdown_pos=torch.tensor([[[0.0, 3.0, 4.0]]]),
        )

        self.assertTrue(torch.allclose(metrics["root_xy_error_mean"], torch.tensor(2.2360679), atol=1e-5))
        self.assertTrue(torch.allclose(metrics["root_z_error_mean"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(metrics["root_yaw_error_mean"], torch.tensor(0.2)))
        self.assertTrue(torch.allclose(metrics["joint_error_mean"], torch.tensor(1.5)))
        self.assertTrue(torch.allclose(metrics["foot_pos_root_error_mean"], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(metrics["contact_match_rate"], torch.tensor(0.75)))
        self.assertTrue(torch.allclose(metrics["touchdown_error_mean"], torch.tensor(5.0)))
        self.assertIn("trajectory_tracking_score", metrics)
        self.assertGreater(metrics["trajectory_tracking_score"].item(), 0.0)

    def test_exponential_tracking_reward_increases_when_error_is_small(self):
        from Go2Pvcnn.extension.mdp.rewards_reference import (
            compare_reference_tensors,
            exponential_tracking_reward,
        )

        low_error = torch.tensor([0.1, 0.2])
        high_error = torch.tensor([1.0, 2.0])

        low_reward = exponential_tracking_reward(low_error, sigma=0.5)
        high_reward = exponential_tracking_reward(high_error, sigma=0.5)

        self.assertTrue(torch.all(low_reward > high_reward))
        self.assertTrue(torch.all(low_reward <= 1.0))

        comparisons = compare_reference_tensors(
            current_root_pos=torch.tensor([[0.0, 0.0, 0.0]]),
            reference_root_pos=torch.tensor([[1.0, 0.0, 0.0]]),
            current_root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            reference_root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            current_joint_pos=torch.tensor([[0.0, 1.0]]),
            reference_joint_pos=torch.tensor([[1.0, 1.0]]),
            current_foot_pos_root=torch.tensor([[[0.0, 0.0, 0.0]]]),
            reference_foot_pos_root=torch.tensor([[[0.0, 2.0, 0.0]]]),
            current_contact_state=torch.tensor([[1.0, 0.0]]),
            reference_contact_state=torch.tensor([[1.0, 1.0]]),
            current_touchdown_pos_w=torch.tensor([[[0.0, 0.0, 0.0]]]),
            reference_touchdown_pos_w=torch.tensor([[[0.0, 0.0, 3.0]]]),
        )
        self.assertTrue(torch.allclose(comparisons["root_position_error"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(comparisons["joint_pos_error"], torch.tensor([0.5])))
        self.assertTrue(torch.allclose(comparisons["foot_pos_root_error"], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(comparisons["contact_state_error"], torch.tensor([0.5])))
        self.assertTrue(torch.allclose(comparisons["touchdown_pos_w_error"], torch.tensor([3.0])))

    def test_reference_comparison_helpers_return_expected_errors(self):
        from Go2Pvcnn.extension.mdp.rewards_reference import (
            compare_contact_state,
            compare_foot_pos_root,
            compare_joint_pos,
            compare_root_state,
            compare_touchdown_pos_w,
            compare_reference_tensors,
        )

        current_root_pos = torch.tensor([[1.0, 2.0, 3.0]])
        ref_root_pos = torch.tensor([[2.0, 4.0, 6.0]])
        current_root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        ref_root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        current_joint_pos = torch.tensor([[0.0, 1.0]])
        ref_joint_pos = torch.tensor([[1.0, 3.0]])
        current_foot_pos = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        ref_foot_pos = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
        current_contact = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        ref_contact = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        current_touchdown = torch.tensor([[[0.0, 0.0, 0.0]]])
        ref_touchdown = torch.tensor([[[0.0, 3.0, 4.0]]])

        root_error = compare_root_state(
            current_root_pos,
            ref_root_pos,
            current_root_quat,
            ref_root_quat,
        )
        self.assertTrue(torch.allclose(root_error["position_error"], torch.tensor([3.7416575]), atol=1e-5))
        self.assertTrue(torch.allclose(root_error["orientation_error"], torch.tensor([0.0])))
        self.assertTrue(torch.allclose(compare_joint_pos(current_joint_pos, ref_joint_pos), torch.tensor([1.5])))
        self.assertTrue(torch.allclose(compare_foot_pos_root(current_foot_pos, ref_foot_pos), torch.tensor([1.0])))
        self.assertTrue(torch.allclose(compare_contact_state(current_contact, ref_contact), torch.tensor([0.25])))
        self.assertTrue(torch.allclose(compare_touchdown_pos_w(current_touchdown, ref_touchdown), torch.tensor([5.0])))

        aggregate = compare_reference_tensors(
            current_root_pos=current_root_pos,
            reference_root_pos=ref_root_pos,
            current_root_quat=current_root_quat,
            reference_root_quat=ref_root_quat,
            current_joint_pos=current_joint_pos,
            reference_joint_pos=ref_joint_pos,
            current_foot_pos_root=current_foot_pos,
            reference_foot_pos_root=ref_foot_pos,
            current_contact_state=current_contact,
            reference_contact_state=ref_contact,
            current_touchdown_pos_w=current_touchdown,
            reference_touchdown_pos_w=ref_touchdown,
        )
        self.assertIn("root_position_error", aggregate)
        self.assertIn("root_orientation_error", aggregate)
        self.assertIn("joint_pos_error", aggregate)
        self.assertIn("foot_pos_root_error", aggregate)
        self.assertIn("contact_state_error", aggregate)
        self.assertIn("touchdown_pos_w_error", aggregate)


if __name__ == "__main__":
    unittest.main()
