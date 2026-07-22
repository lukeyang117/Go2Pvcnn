from __future__ import annotations

import torch

from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule


def test_fixed_trot_schedule_uses_cached_constants_for_cuda_graph_capture() -> None:
    source = __import__("pathlib").Path(
        "Go2Pvcnn/extension/joint_mpc_rti/model/gait_schedule.py"
    ).read_text()

    assert "constant_like" in source
    assert "torch.tensor((0, 12, 12, 0)" not in source


def test_fixed_trot_schedule_returns_b31x4_without_extension_state() -> None:
    phase = torch.tensor([0, 7, 23], dtype=torch.long)

    schedule = fixed_trot_schedule(phase, horizon_steps=30)

    assert schedule.phase.shape == (3, 31, 4)
    assert schedule.swing.shape == (3, 31, 4)
    assert schedule.stance.shape == (3, 31, 4)
    assert schedule.swing_tau.shape == (3, 31, 4)
    assert torch.equal(schedule.swing, ~schedule.stance)
    assert torch.equal(schedule.swing[:, :, 0], schedule.swing[:, :, 3])
    assert torch.equal(schedule.swing[:, :, 1], schedule.swing[:, :, 2])
    assert torch.equal(schedule.swing[:, :, 0], ~schedule.swing[:, :, 1])
    assert not hasattr(schedule, "recovery")
    assert not hasattr(schedule, "extension_age")


def test_every_start_phase_has_exactly_twelve_swing_nodes_per_period() -> None:
    phase = torch.arange(24, dtype=torch.long)

    schedule = fixed_trot_schedule(phase, horizon_steps=23)

    assert torch.equal(schedule.swing.sum(dim=1), torch.full((24, 4), 12))
    assert torch.equal(schedule.phase[:, 1:], (schedule.phase[:, :-1] + 1) % 24)


def test_swing_tau_runs_from_zero_to_one_for_each_leg() -> None:
    schedule = fixed_trot_schedule(torch.tensor([0]), horizon_steps=23)

    for leg in range(4):
        tau = schedule.swing_tau[0, :, leg][schedule.swing[0, :, leg]]
        torch.testing.assert_close(tau, torch.linspace(0.0, 1.0, 12))
