import unittest

import numpy as np


class PlannerCoreTrajectoryTest(unittest.TestCase):
    def test_core_package_exports_minimal_trajectory_entrypoints(self):
        from Go2Pvcnn.extension.planner import core

        self.assertTrue(core.command_is_standstill(core.Command()))
        self.assertIn("trot", core.GAIT_PARAMS)
        self.assertEqual(core.JOINT_LIMITS.shape, (12, 2))

    def test_default_initial_state_and_standstill_trajectory(self):
        from Go2Pvcnn.extension.planner.core import (
            Command,
            HIP_HEIGHT,
            default_initial_state,
            generate_trajectory,
        )

        state = default_initial_state(None, x=0.0, y=0.0)
        self.assertTrue(np.allclose(state.root_pos, np.array([0.0, 0.0, HIP_HEIGHT])))
        self.assertEqual(state.root_quat.shape, (4,))
        self.assertEqual(state.foot_pos.shape, (4, 3))
        self.assertEqual(state.joint_angles.shape, (12,))
        self.assertTrue(np.allclose(state.foot_pos[:, 2], 0.0))

        result = generate_trajectory(None, state, Command(), n_frames=6, dt=0.02)

        self.assertEqual(result.root_pos_w.shape, (6, 3))
        self.assertEqual(result.root_quat_w.shape, (6, 4))
        self.assertEqual(result.contact_state.shape, (6, 4))
        self.assertTrue(np.all(result.contact_state == 1.0))
        self.assertTrue(np.allclose(result.root_pos_w, state.root_pos.reshape(1, 3)))
        self.assertTrue(np.allclose(result.root_quat_w, state.root_quat.reshape(1, 4)))
        self.assertTrue(np.allclose(result.root_lin_vel_w, 0.0))
        self.assertTrue(np.allclose(result.root_ang_vel_w, 0.0))
        self.assertTrue(np.allclose(result.planned_touchdown_w, state.foot_pos))
        self.assertTrue(np.allclose(result.foot_pos_w, state.foot_pos.reshape(1, 4, 3)))
        self.assertEqual(result.body_pos_root.shape, (6, 12, 3))

    def test_non_standstill_command_is_explicitly_unimplemented(self):
        from Go2Pvcnn.extension.planner.core import (
            Command,
            default_initial_state,
            generate_trajectory,
        )

        state = default_initial_state(None, x=0.0, y=0.0)

        with self.assertRaises(NotImplementedError):
            generate_trajectory(None, state, Command(vx=0.1), n_frames=6, dt=0.02)

    def test_gait_and_touchdown_helpers_remain_import_safe(self):
        from Go2Pvcnn.extension.planner.core import gait_schedule, next_touchdown_times, stance_time

        contact_seq = gait_schedule(
            0.0,
            3,
            0.5,
            1.0,
            0.75,
            np.array([0.0, 0.5, 0.75, 0.25], dtype=np.float64),
        )
        self.assertEqual(contact_seq.shape, (3, 4))
        self.assertTrue(np.all((contact_seq == 0.0) | (contact_seq == 1.0)))
        self.assertAlmostEqual(stance_time(2.0, 0.5), 0.25)
        self.assertTrue(
            np.allclose(
                next_touchdown_times(2.0, np.array([0.0, 0.5, 0.25, 0.75], dtype=np.float64)),
                np.array([0.5, 0.25, 0.375, 0.125], dtype=np.float64),
            )
        )


if __name__ == "__main__":
    unittest.main()
