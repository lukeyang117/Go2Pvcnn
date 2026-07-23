"""Fixed tensor schedule for the 24-frame diagonal trot."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.joint_mpc_rti.tensor_constants import constant_like


@dataclass(frozen=True)
class FixedTrotSchedule:
    # Temporary node-mask aliases for modules migrated in later plan tasks.
    phase: Tensor
    swing: Tensor
    stance: Tensor
    swing_tau: Tensor
    phase_node: Tensor | None = None
    phase_edge: Tensor | None = None
    swing_edge: Tensor | None = None
    stance_edge: Tensor | None = None
    stance_node: Tensor | None = None
    touchdown_edge: Tensor | None = None
    liftoff_edge: Tensor | None = None
    swing_tau_node: Tensor | None = None
    steps_to_touchdown_node: Tensor | None = None

    def __post_init__(self) -> None:
        phase_node = self.phase if self.phase_node is None else self.phase_node
        phase_edge = phase_node[:, :-1] if self.phase_edge is None else self.phase_edge
        swing_edge = phase_edge < 12 if self.swing_edge is None else self.swing_edge
        stance_edge = ~swing_edge if self.stance_edge is None else self.stance_edge
        stance_node = self.stance if self.stance_node is None else self.stance_node
        touchdown_edge = (
            (phase_node[:, :-1] == 11) & (phase_node[:, 1:] == 12)
            if self.touchdown_edge is None
            else self.touchdown_edge
        )
        liftoff_edge = (
            (phase_node[:, :-1] == 23) & (phase_node[:, 1:] == 0)
            if self.liftoff_edge is None
            else self.liftoff_edge
        )
        swing_tau_node = self.swing_tau if self.swing_tau_node is None else self.swing_tau_node
        steps_to_touchdown_node = (
            torch.remainder(12 - phase_node, 24)
            if self.steps_to_touchdown_node is None
            else self.steps_to_touchdown_node
        )
        object.__setattr__(self, "phase_node", phase_node)
        object.__setattr__(self, "phase_edge", phase_edge)
        object.__setattr__(self, "swing_edge", swing_edge)
        object.__setattr__(self, "stance_edge", stance_edge)
        object.__setattr__(self, "stance_node", stance_node)
        object.__setattr__(self, "touchdown_edge", touchdown_edge)
        object.__setattr__(self, "liftoff_edge", liftoff_edge)
        object.__setattr__(self, "swing_tau_node", swing_tau_node)
        object.__setattr__(self, "steps_to_touchdown_node", steps_to_touchdown_node)


def fixed_trot_schedule(phase0: Tensor, *, horizon_steps: int = 30) -> FixedTrotSchedule:
    """Broadcast a 12-swing/12-stance diagonal trot from each batch phase."""
    phase0 = torch.as_tensor(phase0, dtype=torch.long)
    if phase0.ndim == 0:
        phase0 = phase0.unsqueeze(0)
    if phase0.ndim != 1:
        raise ValueError("phase0 must have shape [B]")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")

    node = constant_like(
        phase0,
        f"gait_node_index_h{horizon_steps}",
        tuple(range(horizon_steps + 1)),
    )
    leg_offset = constant_like(phase0, "gait_leg_offsets", (0, 12, 12, 0))
    phase_node = (
        phase0[:, None, None] + node[None, :, None] + leg_offset[None, None, :]
    ) % 24
    phase_edge = phase_node[:, :-1]
    swing_edge = phase_edge < 12
    stance_edge = ~swing_edge
    swing_node = phase_node < 12
    stance_node = ~swing_node
    touchdown_edge = (phase_node[:, :-1] == 11) & (phase_node[:, 1:] == 12)
    liftoff_edge = (phase_node[:, :-1] == 23) & (phase_node[:, 1:] == 0)
    phase_float = phase_node.to(torch.float32)
    swing_tau_node = torch.where(
        phase_node <= 12,
        phase_float / 12.0,
        torch.zeros_like(phase_float),
    )
    return FixedTrotSchedule(
        phase_node=phase_node,
        phase_edge=phase_edge,
        swing_edge=swing_edge,
        stance_edge=stance_edge,
        stance_node=stance_node,
        touchdown_edge=touchdown_edge,
        liftoff_edge=liftoff_edge,
        swing_tau_node=swing_tau_node,
        steps_to_touchdown_node=torch.remainder(12 - phase_node, 24),
        phase=phase_node,
        swing=swing_node,
        stance=stance_node,
        swing_tau=swing_tau_node,
    )


__all__ = ["FixedTrotSchedule", "fixed_trot_schedule"]
