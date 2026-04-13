import unittest
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extension.planner.runtime.raw_go2fp_bridge import ensure_kinematic_footsteps_on_syspath


ensure_kinematic_footsteps_on_syspath()


def _pad_event_indices(
    events_by_leg,
    *,
    batch_size: int,
    n_legs: int,
    width: int,
    sentinel: int | None = None,
) -> torch.Tensor:
    sentinel = width + 1 if sentinel is None else sentinel
    padded = torch.full((batch_size, n_legs, width), sentinel, dtype=torch.int64)
    for batch_idx, events in enumerate(events_by_leg):
        for leg in range(n_legs):
            values = torch.tensor(events[leg], dtype=torch.int64)
            padded[batch_idx, leg, : values.numel()] = values
    return padded


def _event_valid_mask(events_by_leg, *, batch_size: int, n_legs: int, width: int) -> torch.Tensor:
    mask = torch.zeros((batch_size, n_legs, width), dtype=torch.bool)
    for batch_idx, events in enumerate(events_by_leg):
        for leg in range(n_legs):
            mask[batch_idx, leg, : len(events[leg])] = True
    return mask


class BatchedGaitTest(unittest.TestCase):
    def test_scalar_inputs_broadcast_and_singleton_inputs(self):
        from extension.batched_planner.gait import (
            batched_gait_schedule,
            batched_legs_requiring_touchdown,
            batched_next_touchdown_times,
            batched_stance_time,
        )
        from scripts.go2fp.gait import gait_schedule as raw_gait_schedule

        contact = batched_gait_schedule(
            0.0,
            4,
            0.02,
            2.0,
            0.55,
            torch.tensor([0.0, 0.5, 0.5, 0.0], dtype=torch.float64),
        )

        self.assertEqual(tuple(contact.shape), (1, 4, 4))
        torch.testing.assert_close(
            contact[0],
            torch.as_tensor(
                raw_gait_schedule(
                    0.0,
                    4,
                    0.02,
                    2.0,
                    0.55,
                    np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float64),
                )
            ),
        )

        touchdown_times = batched_next_touchdown_times(
            2.0,
            torch.tensor([[0.0, 0.5, 0.25, 0.75]], dtype=torch.float64),
        )
        self.assertEqual(tuple(touchdown_times.shape), (1, 4))
        self.assertEqual(touchdown_times.device.type, "cpu")

        stance = batched_stance_time(2.0, 0.55)
        self.assertEqual(tuple(stance.shape), (1,))
        self.assertAlmostEqual(stance.item(), 0.275)

        touchdown_mask = batched_legs_requiring_touchdown(contact)
        self.assertEqual(tuple(touchdown_mask.shape), (1, 4))
        self.assertTrue(torch.equal(touchdown_mask[0], torch.tensor([False, False, False, False])))

    def test_device_preservation_on_non_default_device(self):
        from extension.batched_planner.gait import (
            batched_gait_schedule,
            batched_next_touchdown_times,
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("meta")
        phase_offsets = torch.tensor([0.0, 0.5, 0.75, 0.25], dtype=torch.float64, device=device)

        contact = batched_gait_schedule(0.0, 3, 0.02, 2.0, 0.55, phase_offsets)
        touchdown_times = batched_next_touchdown_times(2.0, phase_offsets)

        self.assertEqual(contact.device.type, device.type)
        self.assertEqual(touchdown_times.device.type, device.type)
        self.assertEqual(tuple(contact.shape), (1, 3, 4))
        self.assertEqual(tuple(touchdown_times.shape), (1, 4))

    def test_mismatched_batch_dimensions_raise_value_error(self):
        from extension.batched_planner.gait import batched_gait_schedule

        with self.assertRaisesRegex(ValueError, "common leading dimension"):
            batched_gait_schedule(
                torch.tensor([0.0, 0.1], dtype=torch.float64),
                3,
                0.02,
                torch.tensor([2.0], dtype=torch.float64),
                torch.tensor([0.55, 0.60, 0.65], dtype=torch.float64),
                torch.tensor([[0.0, 0.5, 0.5, 0.0]], dtype=torch.float64),
            )

    def test_mixed_device_inputs_raise_value_error(self):
        from extension.batched_planner.gait import batched_next_touchdown_times

        phase_offsets = torch.tensor([0.0, 0.5, 0.25, 0.75], dtype=torch.float64, device="meta")
        step_freq = torch.tensor(2.0, dtype=torch.float64)

        with self.assertRaisesRegex(ValueError, "multiple devices"):
            batched_next_touchdown_times(step_freq, phase_offsets)

    def test_gait_schedule_matches_raw(self):
        from extension.batched_planner.gait import batched_gait_schedule
        from scripts.go2fp.gait import gait_schedule as raw_gait_schedule

        t0 = torch.tensor([0.0, 0.13], dtype=torch.float64)
        dt = 0.02
        step_freq = torch.tensor([2.0, 1.0], dtype=torch.float64)
        duty_factor = torch.tensor([0.55, 0.75], dtype=torch.float64)
        phase_offsets = torch.tensor(
            [
                [0.0, 0.5, 0.5, 0.0],
                [0.0, 0.5, 0.75, 0.25],
            ],
            dtype=torch.float64,
        )

        actual = batched_gait_schedule(t0, 6, dt, step_freq, duty_factor, phase_offsets)

        self.assertEqual(tuple(actual.shape), (2, 6, 4))
        self.assertEqual(actual.dtype, torch.float32)
        for idx in range(2):
            expected = raw_gait_schedule(
                float(t0[idx].item()),
                6,
                dt,
                float(step_freq[idx].item()),
                float(duty_factor[idx].item()),
                phase_offsets[idx].cpu().numpy(),
            )
            torch.testing.assert_close(actual[idx], torch.as_tensor(expected))

    def test_next_touchdown_times_matches_raw(self):
        from extension.batched_planner.gait import batched_next_touchdown_times
        from scripts.go2fp.gait import next_touchdown_times as raw_next_touchdown_times

        step_freq = torch.tensor([2.0, 1.5], dtype=torch.float64)
        phase_offsets = torch.tensor(
            [
                [0.0, 0.5, 0.25, 0.75],
                [0.0, 0.05, 0.4, 0.35],
            ],
            dtype=torch.float64,
        )

        actual = batched_next_touchdown_times(step_freq, phase_offsets)

        self.assertEqual(tuple(actual.shape), (2, 4))
        for idx in range(2):
            expected = raw_next_touchdown_times(
                float(step_freq[idx].item()),
                phase_offsets[idx].cpu().numpy(),
            )
            torch.testing.assert_close(actual[idx], torch.as_tensor(expected))

    def test_stance_time_matches_raw(self):
        from extension.batched_planner.gait import batched_stance_time
        from scripts.go2fp.gait import stance_time as raw_stance_time

        step_freq = torch.tensor([2.0, 0.5], dtype=torch.float64)
        duty_factor = torch.tensor([0.55, 0.80], dtype=torch.float64)

        actual = batched_stance_time(step_freq, duty_factor)

        self.assertEqual(tuple(actual.shape), (2,))
        for idx in range(2):
            expected = raw_stance_time(float(step_freq[idx].item()), float(duty_factor[idx].item()))
            self.assertAlmostEqual(actual[idx].item(), expected)

    def test_legs_requiring_touchdown_matches_raw(self):
        from extension.batched_planner.gait import batched_legs_requiring_touchdown
        from scripts.go2fp.foothold import legs_requiring_touchdown as raw_legs_requiring_touchdown

        contact_seq = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )

        actual = batched_legs_requiring_touchdown(contact_seq)

        self.assertEqual(tuple(actual.shape), (2, 4))
        for idx in range(2):
            expected = raw_legs_requiring_touchdown(contact_seq[idx].cpu().numpy())
            np.testing.assert_array_equal(actual[idx].cpu().numpy(), expected)

    def test_legs_requiring_touchdown_preserves_batch_dim_for_unbatched_input(self):
        from extension.batched_planner.gait import batched_legs_requiring_touchdown

        contact_seq = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        actual = batched_legs_requiring_touchdown(contact_seq)

        self.assertEqual(tuple(actual.shape), (1, 4))
        self.assertTrue(torch.equal(actual[0], torch.tensor([True, False, False, True])))

    def test_detect_swing_events_matches_raw(self):
        from extension.batched_planner.gait import batched_detect_swing_events
        from scripts.go2fp.gait import detect_swing_events as raw_detect_swing_events

        contact_seq = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )

        actual = batched_detect_swing_events(contact_seq)

        self.assertIsInstance(actual, dict)
        self.assertEqual(tuple(actual["lift_off"].shape), (2, 4, 4))
        self.assertEqual(tuple(actual["touch_down"].shape), (2, 4, 4))
        self.assertEqual(tuple(actual["lift_off_valid"].shape), (2, 4, 4))
        self.assertEqual(tuple(actual["touch_down_valid"].shape), (2, 4, 4))
        self.assertTrue(torch.all(actual["lift_off"][~actual["lift_off_valid"]] == 5))
        self.assertTrue(torch.all(actual["touch_down"][~actual["touch_down_valid"]] == 5))

        raw_events = [raw_detect_swing_events(contact_seq[idx].cpu().numpy()) for idx in range(2)]
        expected_lift_off = _pad_event_indices(
            [events["lift_off"] for events in raw_events],
            batch_size=2,
            n_legs=4,
            width=4,
        )
        expected_touch_down = _pad_event_indices(
            [events["touch_down"] for events in raw_events],
            batch_size=2,
            n_legs=4,
            width=4,
        )
        expected_lift_off_valid = _event_valid_mask(
            [events["lift_off"] for events in raw_events],
            batch_size=2,
            n_legs=4,
            width=4,
        )
        expected_touch_down_valid = _event_valid_mask(
            [events["touch_down"] for events in raw_events],
            batch_size=2,
            n_legs=4,
            width=4,
        )

        torch.testing.assert_close(actual["lift_off"], expected_lift_off)
        torch.testing.assert_close(actual["touch_down"], expected_touch_down)
        torch.testing.assert_close(actual["lift_off_valid"], expected_lift_off_valid)
        torch.testing.assert_close(actual["touch_down_valid"], expected_touch_down_valid)

    def test_detect_swing_events_uses_sequence_length_sentinel_for_terminal_frame_events(self):
        from extension.batched_planner.gait import batched_detect_swing_events

        contact_seq = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        actual = batched_detect_swing_events(contact_seq)

        self.assertEqual(tuple(actual["touch_down"].shape), (1, 4, 2))
        self.assertTrue(torch.equal(actual["touch_down"][0, 0], torch.tensor([2, 3])))
        self.assertTrue(torch.equal(actual["touch_down_valid"][0, 0], torch.tensor([True, False])))
        self.assertTrue(torch.equal(actual["lift_off"][0, 0], torch.tensor([1, 3])))
        self.assertTrue(torch.equal(actual["lift_off_valid"][0, 0], torch.tensor([True, False])))

    def test_detect_swing_events_handles_unbatched_input(self):
        from extension.batched_planner.gait import batched_detect_swing_events
        from scripts.go2fp.gait import detect_swing_events as raw_detect_swing_events

        contact_seq = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        actual = batched_detect_swing_events(contact_seq)

        expected = raw_detect_swing_events(contact_seq.cpu().numpy())
        self.assertEqual(tuple(actual["lift_off"].shape), (1, 4, 4))
        self.assertEqual(tuple(actual["touch_down"].shape), (1, 4, 4))
        self.assertEqual(tuple(actual["lift_off_valid"].shape), (1, 4, 4))
        self.assertEqual(tuple(actual["touch_down_valid"].shape), (1, 4, 4))
        torch.testing.assert_close(
            actual["lift_off"],
            _pad_event_indices([expected["lift_off"]], batch_size=1, n_legs=4, width=4),
        )
        torch.testing.assert_close(
            actual["touch_down"],
            _pad_event_indices([expected["touch_down"]], batch_size=1, n_legs=4, width=4),
        )
        torch.testing.assert_close(
            actual["lift_off_valid"],
            _event_valid_mask([expected["lift_off"]], batch_size=1, n_legs=4, width=4),
        )
        torch.testing.assert_close(
            actual["touch_down_valid"],
            _event_valid_mask([expected["touch_down"]], batch_size=1, n_legs=4, width=4),
        )

    def test_detect_swing_events_handles_short_sequences_and_empty_events(self):
        from extension.batched_planner.gait import batched_detect_swing_events

        for contact_seq, expected_shape in (
            (torch.empty((0, 4), dtype=torch.float32), (1, 4, 0)),
            (torch.ones((1, 4), dtype=torch.float32), (1, 4, 0)),
        ):
            actual = batched_detect_swing_events(contact_seq)

            self.assertEqual(tuple(actual["lift_off"].shape), expected_shape)
            self.assertEqual(tuple(actual["touch_down"].shape), expected_shape)
            self.assertEqual(tuple(actual["lift_off_valid"].shape), expected_shape)
            self.assertEqual(tuple(actual["touch_down_valid"].shape), expected_shape)
            self.assertEqual(actual["lift_off"].numel(), 0)
            self.assertEqual(actual["touch_down"].numel(), 0)
            self.assertEqual(actual["lift_off_valid"].numel(), 0)
            self.assertEqual(actual["touch_down_valid"].numel(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_detect_swing_events_preserves_cuda_device(self):
        from extension.batched_planner.gait import batched_detect_swing_events

        contact_seq = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                ]
            ],
            dtype=torch.float32,
            device="cuda",
        )

        actual = batched_detect_swing_events(contact_seq)

        self.assertEqual(actual["lift_off"].device.type, "cuda")
        self.assertEqual(actual["touch_down"].device.type, "cuda")
        self.assertEqual(actual["lift_off_valid"].device.type, "cuda")
        self.assertEqual(actual["touch_down_valid"].device.type, "cuda")

    def test_detect_swing_events_tracks_multiple_events_per_leg(self):
        from extension.batched_planner.gait import batched_detect_swing_events

        contact_seq = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        actual = batched_detect_swing_events(contact_seq)

        self.assertEqual(tuple(actual["lift_off"].shape), (1, 4, 4))
        self.assertEqual(tuple(actual["touch_down"].shape), (1, 4, 4))
        self.assertTrue(torch.equal(actual["lift_off"][0, 0], torch.tensor([1, 3, 5, 5])))
        self.assertTrue(torch.equal(actual["touch_down"][0, 0], torch.tensor([2, 4, 5, 5])))
        self.assertTrue(torch.equal(actual["lift_off_valid"][0, 0], torch.tensor([True, True, False, False])))
        self.assertTrue(torch.equal(actual["touch_down_valid"][0, 0], torch.tensor([True, True, False, False])))

    def test_detect_swing_events_compacts_valid_indices_in_ascending_order(self):
        from extension.batched_planner.gait import batched_detect_swing_events

        contact_seq = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )

        actual = batched_detect_swing_events(contact_seq)

        sentinel = contact_seq.shape[1]
        for key in ("lift_off", "touch_down"):
            values = actual[key]
            valid = actual[f"{key}_valid"]
            self.assertEqual(tuple(values.shape), (2, 4, 6))
            self.assertTrue(torch.all(values[~valid] == sentinel))
            for batch_idx in range(values.shape[0]):
                for leg in range(values.shape[1]):
                    valid_count = int(valid[batch_idx, leg].sum().item())
                    expected_valid = torch.tensor(
                        [True] * valid_count + [False] * (values.shape[-1] - valid_count),
                        dtype=torch.bool,
                    )
                    self.assertTrue(torch.equal(valid[batch_idx, leg], expected_valid))
                    if valid_count > 1:
                        self.assertTrue(torch.all(values[batch_idx, leg, :valid_count - 1] < values[batch_idx, leg, 1:valid_count]))

    def test_detect_swing_events_handles_batched_short_sequences_and_empty_events(self):
        from extension.batched_planner.gait import batched_detect_swing_events

        for contact_seq in (
            torch.empty((2, 0, 4), dtype=torch.float32),
            torch.ones((2, 1, 4), dtype=torch.float32),
        ):
            actual = batched_detect_swing_events(contact_seq)

            self.assertEqual(tuple(actual["lift_off"].shape), (2, 4, 0))
            self.assertEqual(tuple(actual["touch_down"].shape), (2, 4, 0))
            self.assertEqual(tuple(actual["lift_off_valid"].shape), (2, 4, 0))
            self.assertEqual(tuple(actual["touch_down_valid"].shape), (2, 4, 0))
            self.assertEqual(actual["lift_off"].numel(), 0)
            self.assertEqual(actual["touch_down"].numel(), 0)
            self.assertEqual(actual["lift_off_valid"].numel(), 0)
            self.assertEqual(actual["touch_down_valid"].numel(), 0)


if __name__ == "__main__":
    unittest.main()
