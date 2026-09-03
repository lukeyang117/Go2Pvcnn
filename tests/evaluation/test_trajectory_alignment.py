from __future__ import annotations

import torch

from evaluation.trajectory_alignment import ValidWindowBuffer, tracking_mse


def test_alignment_rejects_plan_id_change_and_reset() -> None:
    buffer = ValidWindowBuffer(1, horizon=24, device=torch.device("cpu"))
    frame = torch.zeros(1, 39)
    final = torch.zeros(1, dtype=torch.bool)
    for step in range(24):
        final = buffer.push(
            frame,
            frame,
            torch.tensor([0 if step < 23 else 1]),
            torch.tensor([True]),
            torch.tensor([step == 12]),
            torch.tensor([True]),
        )
    assert not bool(final.item())


def test_tracking_mse_anchors_root_to_window_start() -> None:
    actual = torch.zeros(1, 24, 39)
    reference = actual.clone()
    actual[:, :, 24] = torch.arange(24, dtype=torch.float32)
    reference[:, :, 24] = torch.arange(24, dtype=torch.float32) + 0.1 * torch.arange(24, dtype=torch.float32)
    valid = torch.ones(1, 24, dtype=torch.bool)
    metrics = tracking_mse(actual, reference, valid, actual[:, :1, 24:27])
    expected = ((0.1 * torch.arange(24, dtype=torch.float32)).square().mean() / 3.0).item()
    assert metrics["root_position_mse"].item() == expected


def test_alignment_emits_first_complete_window_after_23_transitions() -> None:
    buffer = ValidWindowBuffer(1, horizon=24, device=torch.device("cpu"))
    frame = torch.zeros(1, 39)
    flags = []
    for step in range(23):
        flags.append(bool(buffer.push(frame, frame, torch.tensor([2]), torch.tensor([True]), torch.tensor([False]), torch.tensor([True])).item()))
    assert flags[-1] is True
