"""Fixed-shape kinematic rollout and derived Go2 geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch._higher_order_ops.scan import scan

from extension.joint_mpc_rti.model.dynamics import kinematic_step
from extension.joint_mpc_rti.model.go2_kinematics import go2_fk
from extension.joint_mpc_rti.types import JointMpcRtiState


@dataclass(frozen=True)
class JointMpcRollout:
    state: Tensor
    control: Tensor
    foot_pos_w: Tensor
    knee_pos_w: Tensor
    shank_samples_w: Tensor
    body_samples_w: Tensor


def _fixed_horizon_state_rollout(state0: Tensor, controls: Tensor, dt: float) -> Tensor:
    state = state0
    sequence: list[Tensor] = [state0]
    for index in range(int(controls.shape[1])):
        state = kinematic_step(state, controls[:, index], dt=dt)
        sequence.append(state)
    return torch.stack(sequence, dim=1)


_COMPILED_FIXED_HORIZON_STATE_ROLLOUT = torch.compile(
    _fixed_horizon_state_rollout,
    fullgraph=True,
    dynamic=False,
    options={"triton.cudagraphs": False},
)


def rollout_state_sequence(
    initial_state: JointMpcRtiState,
    control: Tensor,
    *,
    dt: float,
    compile_kernels: bool = False,
) -> Tensor:
    controls = torch.as_tensor(control, dtype=initial_state.root_pos_w.dtype, device=initial_state.device)
    state0 = initial_state.as_vector()
    if bool(compile_kernels) and state0.is_cuda:
        return _COMPILED_FIXED_HORIZON_STATE_ROLLOUT(state0, controls, float(dt))

    def combine(previous: Tensor, control_step: Tensor) -> tuple[Tensor, Tensor]:
        next_state = kinematic_step(previous, control_step, dt=dt)
        return next_state, next_state

    _, scanned_state = scan(combine, state0, controls.transpose(0, 1), dim=0)
    return torch.cat((state0.unsqueeze(1), scanned_state.transpose(0, 1)), dim=1)


def rollout_controls(
    initial_state: JointMpcRtiState,
    control: Tensor,
    *,
    dt: float,
    compile_kernels: bool = False,
) -> JointMpcRollout:
    """Roll out controls with a graph-level scan and derive all collision geometry through FK."""
    controls = torch.as_tensor(control, dtype=initial_state.root_pos_w.dtype, device=initial_state.device)
    if controls.ndim != 3 or int(controls.shape[0]) != initial_state.batch_size or int(controls.shape[-1]) != 18:
        raise ValueError("control must have shape [B,H,18]")
    state = rollout_state_sequence(
        initial_state,
        controls,
        dt=dt,
        compile_kernels=compile_kernels,
    )
    batch, nodes = int(state.shape[0]), int(state.shape[1])
    geometry = go2_fk(
        state[:, :, :3].reshape(batch * nodes, 3),
        state[:, :, 3:6].reshape(batch * nodes, 3),
        state[:, :, 6:].reshape(batch * nodes, 12),
    )
    return JointMpcRollout(
        state=state,
        control=controls,
        foot_pos_w=geometry.foot_pos_w.reshape(batch, nodes, 4, 3),
        knee_pos_w=geometry.knee_pos_w.reshape(batch, nodes, 4, 3),
        shank_samples_w=geometry.shank_samples_w.reshape(batch, nodes, 4, 3, 3),
        body_samples_w=geometry.body_samples_w.reshape(batch, nodes, geometry.body_samples_w.shape[1], 3),
    )


__all__ = ["JointMpcRollout", "rollout_controls", "rollout_state_sequence"]
