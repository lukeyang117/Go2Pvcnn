from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [
        (0.0, (1.0, 0.0)),
        (torch.pi / 2, (0.0, 1.0)),
        (torch.pi, (-1.0, 0.0)),
    ],
)
def test_body_forward_command_rotates_once_into_world(yaw: float, expected: tuple[float, float]) -> None:
    from extension.joint_mpc_rti.integration.command import body_linear_velocity_to_world

    command = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    actual = body_linear_velocity_to_world(command, torch.tensor([yaw]))
    torch.testing.assert_close(actual, torch.tensor([expected]), atol=1.0e-5, rtol=0.0)


def test_kinematic_step_integrates_body_velocity_and_joint_velocity() -> None:
    from extension.joint_mpc_rti.model.dynamics import kinematic_step

    state = torch.zeros(1, 18)
    state[:, 5] = torch.pi / 2
    control = torch.zeros(1, 18)
    control[:, 0] = 1.0
    control[:, 6:] = 0.5

    actual = kinematic_step(state, control, dt=0.02)

    torch.testing.assert_close(actual[:, :2], torch.tensor([[0.0, 0.02]]), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(actual[:, 6:], torch.full((1, 12), 0.01))


def test_fixed_trot_uses_diagonal_pairs_and_has_no_optimized_phase() -> None:
    from extension.joint_mpc_rti.model.gait_schedule import fixed_trot_schedule

    contact = fixed_trot_schedule(batch=2, horizon_steps=16, device="cpu")

    assert contact.shape == (2, 17, 4)
    assert contact.dtype == torch.bool
    assert torch.equal(contact[:, :, 0], contact[:, :, 3])
    assert torch.equal(contact[:, :, 1], contact[:, :, 2])
    assert torch.equal(contact[:, :, 0], torch.logical_not(contact[:, :, 1]))
