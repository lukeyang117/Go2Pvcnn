from __future__ import annotations

import pytest
import torch

from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule


def test_fixed_trot_schedule_uses_cached_constants_for_cuda_graph_capture() -> None:
    source = __import__("pathlib").Path(
        "Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py"
    ).read_text()

    assert "constant_like" in source
    assert "torch.tensor((0, 12, 12, 0)" not in source


def test_fixed_trot_schedule_separates_31_nodes_from_30_edges() -> None:
    phase = torch.tensor([0, 7, 23], dtype=torch.long)

    schedule = fixed_trot_schedule(phase, horizon_steps=30)

    assert schedule.phase_node.shape == (3, 31, 4)
    assert schedule.phase_edge.shape == (3, 30, 4)
    assert schedule.swing_edge.shape == (3, 30, 4)
    assert schedule.stance_edge.shape == (3, 30, 4)
    assert schedule.touchdown_edge.shape == (3, 30, 4)
    assert schedule.liftoff_edge.shape == (3, 30, 4)
    assert schedule.stance_node.shape == (3, 31, 4)
    assert schedule.swing_tau_node.shape == (3, 31, 4)
    assert schedule.steps_to_touchdown_node.shape == (3, 31, 4)
    assert schedule.swing.shape == (3, 31, 4)
    assert schedule.stance.shape == (3, 31, 4)
    assert torch.equal(schedule.swing, ~schedule.stance)
    assert torch.equal(schedule.swing[:, :, 0], schedule.swing[:, :, 3])
    assert torch.equal(schedule.swing[:, :, 1], schedule.swing[:, :, 2])
    assert torch.equal(schedule.swing[:, :, 0], ~schedule.swing[:, :, 1])
    assert not hasattr(schedule, "recovery")
    assert not hasattr(schedule, "extension_age")


def test_phase_11_to_12_is_touchdown_edge_and_phase_12_is_stance_node() -> None:
    schedule = fixed_trot_schedule(torch.tensor([0]), horizon_steps=30)

    assert schedule.swing_edge[0, 11, 0]
    assert schedule.touchdown_edge[0, 11, 0]
    assert schedule.stance_node[0, 12, 0]
    assert not schedule.swing[0, 12, 0]
    assert schedule.swing_tau_node[0, 11, 0].item() == pytest.approx(11.0 / 12.0)
    assert schedule.swing_tau_node[0, 12, 0].item() == pytest.approx(1.0)
    assert schedule.steps_to_touchdown_node[0, 11, 0].item() == 1
    assert schedule.steps_to_touchdown_node[0, 12, 0].item() == 0
    assert schedule.steps_to_touchdown_node[0, 13, 0].item() == 23


def test_every_start_phase_has_exactly_twelve_swing_edges_per_period() -> None:
    phase = torch.arange(24, dtype=torch.long)

    schedule = fixed_trot_schedule(phase, horizon_steps=24)

    assert torch.equal(schedule.swing_edge.sum(dim=1), torch.full((24, 4), 12))
    assert torch.equal(schedule.stance_edge.sum(dim=1), torch.full((24, 4), 12))
    assert torch.equal(schedule.touchdown_edge.sum(dim=1), torch.ones((24, 4), dtype=torch.long))
    assert torch.equal(schedule.liftoff_edge.sum(dim=1), torch.ones((24, 4), dtype=torch.long))
    assert torch.equal(schedule.phase_node[:, 1:], (schedule.phase_node[:, :-1] + 1) % 24)
    assert torch.equal(schedule.swing_edge[:, :, 0], schedule.swing_edge[:, :, 3])
    assert torch.equal(schedule.swing_edge[:, :, 1], schedule.swing_edge[:, :, 2])
    assert torch.equal(schedule.swing_edge[:, :, 0], ~schedule.swing_edge[:, :, 1])
    assert int(schedule.steps_to_touchdown_node.min().item()) == 0
    assert int(schedule.steps_to_touchdown_node.max().item()) == 23


def test_swing_tau_reaches_one_only_at_touchdown_node() -> None:
    schedule = fixed_trot_schedule(torch.tensor([0]), horizon_steps=23)

    for leg in range(4):
        phase = schedule.phase_node[0, :, leg]
        swing_tau = schedule.swing_tau_node[0, :, leg][phase < 12]
        torch.testing.assert_close(swing_tau, torch.arange(12, dtype=torch.float32) / 12.0)
        touchdown_tau = schedule.swing_tau_node[0, :, leg][phase == 12]
        torch.testing.assert_close(touchdown_tau, torch.ones_like(touchdown_tau))
