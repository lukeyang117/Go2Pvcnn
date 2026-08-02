from __future__ import annotations

import torch

from scripts import play


def test_panel_command_clamps_to_signed_parallelism_limits() -> None:
    state = play.ParallelismPlayPanelState(vx=3.0, vy=-2.0, vyaw=-4.0)
    command = torch.zeros(1, 3, dtype=torch.float32)

    actual = play._panel_command_tensor(state, command)

    torch.testing.assert_close(actual, torch.tensor([[1.0, -0.5, -1.0]]))


def test_suppressed_termination_stays_diagnostic_but_not_done() -> None:
    raw_masks = {
        "base_contact": torch.tensor([True]),
        "parallelism_ref_joint_pos_too_far": torch.tensor([True]),
    }

    done, diagnostics = play._filter_termination_masks(
        raw_masks,
        {"base_contact": True, "parallelism_ref_joint_pos_too_far": False},
    )

    assert diagnostics["base_contact"].item() is True
    assert diagnostics["parallelism_ref_joint_pos_too_far"].item() is True
    assert done.item() is True


def test_all_suppressed_terminations_produce_no_done() -> None:
    raw_masks = {
        "time_out": torch.tensor([True]),
        "bad_orientation": torch.tensor([True]),
    }

    done, _ = play._filter_termination_masks(raw_masks, {"time_out": True, "bad_orientation": True})

    assert done.tolist() == [False]
