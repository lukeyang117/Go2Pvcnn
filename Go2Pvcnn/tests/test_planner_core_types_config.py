import unittest
from pathlib import Path

import numpy as np


class PlannerCoreTypesConfigTest(unittest.TestCase):
    def test_core_types_expose_expected_constants_and_helpers(self):
        from Go2Pvcnn.extension.planner.core import types

        self.assertEqual(types.LEG_ORDER, ("FL", "FR", "RL", "RR"))
        self.assertEqual(types.HIP_OFFSETS_ARRAY.shape, (4, 3))
        self.assertTrue(np.allclose(types.HIP_OFFSETS["FL"], np.array([0.1934, 0.0465, 0.0])))
        self.assertTrue(np.allclose(types.quat_from_yaw(0.0), np.array([1.0, 0.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(types.yaw_rotation_matrix(0.0), np.eye(3)))

        state = types.RobotState(
            root_pos=np.zeros(3),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_angles=np.zeros(12),
            foot_pos=np.zeros((4, 3)),
        )
        self.assertEqual(state.foot_vel.shape, (4, 3))

    def test_core_config_resolves_custom_elevation_path_and_defaults(self):
        from Go2Pvcnn.extension.planner.core.config import PlannerConfig, PlannerTerrainConfig

        custom_path = Path("/tmp/custom_heightmap.npz")
        terrain_cfg = PlannerTerrainConfig(
            heightmap_path=Path("sample.png"),
            elevation_path=custom_path,
        )
        self.assertEqual(terrain_cfg.resolved_elevation_path(), custom_path)

        planner_cfg = PlannerConfig()
        self.assertEqual(planner_cfg.gait_name, "trot")
        self.assertEqual(planner_cfg.trajectory.step_freq, 2.0)
        expected_heightmap = Path(__file__).resolve().parents[1] / "assets" / "terrain" / "sample_heightmap.png"
        self.assertEqual(
            planner_cfg.terrain.heightmap_path,
            expected_heightmap,
        )


if __name__ == "__main__":
    unittest.main()
