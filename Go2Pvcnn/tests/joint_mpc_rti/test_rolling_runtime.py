from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.runtime.manager import JointMpcRtiManager

from .helpers import make_command, make_flat_field, make_state


def test_manager_publishes_x1_and_tracks_full_horizon_from_measured_x0() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=2, device="cpu")
    measured_t = make_state(batch=2)

    step = manager.plan_from_tensors(measured_t, make_command(batch=2), make_flat_field(batch=2))

    torch.testing.assert_close(step.full_trajectory.state[:, 0], measured_t.as_vector())
    torch.testing.assert_close(step.pending_reference.joint_angles, step.full_trajectory.state[:, 1, 6:])
    torch.testing.assert_close(step.pending_reference.root_pos_w, step.full_trajectory.state[:, 1, :3])
    assert step.pending_reference.target_step == 1
    assert torch.all(step.pending_reference.valid)


def test_second_plan_reinjects_new_measured_state_instead_of_old_prediction() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=2, device="cpu")
    first_state = make_state(batch=2)
    manager.plan_from_tensors(first_state, make_command(batch=2), make_flat_field(batch=2))
    second_state = make_state(batch=2)
    second_state.root_pos_w[:, 0] = torch.tensor([0.7, -0.4])

    second = manager.plan_from_tensors(second_state, make_command(batch=2), make_flat_field(batch=2))

    torch.testing.assert_close(second.full_trajectory.state[:, 0], second_state.as_vector())


def test_reset_clears_only_selected_pending_rows() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=4, device="cpu")
    manager.plan_from_tensors(make_state(4), make_command(4), make_flat_field(4))

    manager.reset_envs(torch.tensor([False, True, False, True]))

    assert torch.equal(manager.pending_valid, torch.tensor([True, False, True, False]))


def test_current_reference_is_always_first_future_frame() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=3, device="cpu")
    step = manager.plan_from_tensors(make_state(3), make_command(3), make_flat_field(3))

    current = manager.current_reference()

    torch.testing.assert_close(current["joint_angles"], step.full_trajectory.state[:, 1, 6:])
    assert torch.equal(manager.current_frame_ids(), torch.ones(3, dtype=torch.long))
