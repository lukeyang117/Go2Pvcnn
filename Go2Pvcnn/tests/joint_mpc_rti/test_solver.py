from __future__ import annotations

import torch

from .helpers import make_state


def test_shift_warm_start_injects_measured_x0_and_shifts_controls() -> None:
    from extension.joint_mpc_rti.runtime.warm_start import shift_warm_start

    state = torch.arange(1 * 17 * 18, dtype=torch.float32).reshape(1, 17, 18)
    control = torch.arange(1 * 16 * 18, dtype=torch.float32).reshape(1, 16, 18)
    measured = torch.full((1, 18), -2.0)

    shifted = shift_warm_start(state, control, measured)

    torch.testing.assert_close(shifted.state[:, 0], measured)
    torch.testing.assert_close(shifted.state[:, 1:-1], state[:, 2:])
    torch.testing.assert_close(shifted.control[:, :-1], control[:, 1:])


def test_rollout_geometry_is_derived_from_root_and_joint_only() -> None:
    from extension.joint_mpc_rti.model.rollout import rollout_controls

    result = rollout_controls(make_state(batch=2), torch.zeros(2, 16, 18), dt=0.02)

    assert result.state.shape == (2, 17, 18)
    assert result.foot_pos_w.shape == (2, 17, 4, 3)
    assert result.knee_pos_w.shape == (2, 17, 4, 3)
    assert result.shank_samples_w.shape == (2, 17, 4, 3, 3)
    assert result.body_samples_w.shape[:2] == (2, 17)
    assert not hasattr(result, "independent_foot_state")
