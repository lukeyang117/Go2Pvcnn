from __future__ import annotations

import torch


def test_relaxed_barrier_is_finite_and_increases_toward_violation() -> None:
    from extension.joint_mpc_rti.losses.barriers import relaxed_barrier

    margin = torch.tensor([0.2, 0.02, 0.0, -0.02], requires_grad=True)
    value = relaxed_barrier(margin, relaxation=0.01)

    assert torch.isfinite(value).all()
    assert torch.all(value[1:] > value[:-1])
    value.sum().backward()
    assert torch.isfinite(margin.grad).all()


def test_each_state_trajectory_residual_is_batched_and_finite() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.losses.objective import trajectory_residuals
    from .test_trajectory_losses import _context, _state

    state = _state(batch=3)
    residuals = trajectory_residuals(state, _context(state), JointMpcRtiCfg())

    assert tuple(residuals) == (
        "command",
        "step",
        "contact",
        "swing_speed",
        "terrain",
        "posture",
        "smooth",
    )
    assert all(value.ndim == 2 and value.shape[0] == 3 for value in residuals.values())
    assert all(torch.isfinite(value).all() for value in residuals.values())


def test_terrain_loss_penalizes_low_full_body_geometry() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.losses.terrain import terrain_loss
    from .test_trajectory_losses import _context, _state

    safe = _state(batch=1)
    low = safe.clone()
    low[..., 2] -= 0.25
    context = _context(safe)

    assert terrain_loss(low, context, JointMpcRtiCfg()) > terrain_loss(safe, context, JointMpcRtiCfg())
