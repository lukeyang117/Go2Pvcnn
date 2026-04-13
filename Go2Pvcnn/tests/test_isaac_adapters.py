import unittest


class IsaacAdapterScaffoldingTest(unittest.TestCase):
    def test_state_adapter_helpers_validate_sizes(self):
        from Go2Pvcnn.extension.planner.adapters.isaac_state import (
            IsaacStateAdapterConfig,
            IsaacStateSnapshot,
            normalize_quaternion,
            normalize_vector3,
        )

        self.assertEqual(normalize_vector3([1, 2, 3]), (1.0, 2.0, 3.0))
        self.assertEqual(normalize_quaternion([1, 0, 0, 0]), (1.0, 0.0, 0.0, 0.0))

        config = IsaacStateAdapterConfig()
        self.assertEqual(config.root_frame, "world")
        self.assertEqual(config.foot_frame_names, ("LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"))

        snapshot = IsaacStateSnapshot()
        self.assertFalse(snapshot.is_ready())
        self.assertEqual(snapshot.missing_fields(), ("root_position_w", "root_quaternion_w"))

    def test_heightmap_adapter_reports_shape_and_cell_count(self):
        from Go2Pvcnn.extension.planner.adapters.isaac_heightmap import (
            HeightmapAdapterConfig,
            heightmap_shape,
            is_rectangular_heightmap,
        )

        grid = [[1.0, 2.0], [3.0, 4.0]]
        self.assertTrue(is_rectangular_heightmap(grid))
        self.assertEqual(heightmap_shape(grid), (2, 2))

        config = HeightmapAdapterConfig(resolution_m=0.5, size_m=2.0)
        self.assertEqual(config.cell_count(), 4)

    def test_marker_adapter_builds_payloads(self):
        from Go2Pvcnn.extension.planner.adapters.isaac_markers import (
            IsaacMarkerAdapterConfig,
            MarkerSpec,
            marker_names,
            marker_payloads,
        )

        specs = (
            MarkerSpec(name="goal", position_w=(1.0, 2.0, 3.0), color_rgba=(0.1, 0.2, 0.3, 0.4), scale=0.2),
            MarkerSpec(name="start", position_w=(0.0, 0.0, 0.0)),
        )

        self.assertEqual(marker_names(specs), ("goal", "start"))
        payloads = marker_payloads(specs)
        self.assertEqual(payloads[0]["name"], "goal")
        self.assertEqual(payloads[0]["color_rgba"], (0.1, 0.2, 0.3, 0.4))
        self.assertEqual(payloads[1]["scale"], 0.05)

        config = IsaacMarkerAdapterConfig()
        self.assertEqual(config.normalized_color(), (1.0, 0.4, 0.1, 1.0))


if __name__ == "__main__":
    unittest.main()
