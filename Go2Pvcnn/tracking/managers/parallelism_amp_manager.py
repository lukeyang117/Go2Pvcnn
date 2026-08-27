"""Batched history and state-window construction for Parallelism AMP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AmpStepPayload:
    """Per-environment AMP data emitted after one physics transition."""

    expert_window: Tensor
    agent_window: Tensor
    amp_active: Tensor
    history_ratio: Tensor


def _rotation_6d_to_matrix(value: Tensor) -> Tensor:
    a1, a2 = value[..., :3], value[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1.0e-6)
    a2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1, eps=1.0e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _matrix_to_rotation_6d(value: Tensor) -> Tensor:
    return value[..., :, :2].transpose(-1, -2).reshape(*value.shape[:-2], 6)


class ParallelismAmpManager:
    """Keep paired expert/agent transition rings for a vectorized environment batch.

    The public methods operate on tensors whose first dimension is ``num_envs``.
    No environment row is processed by a Python loop.  A state frame uses the
    fixed 39-dimensional layout documented by the AMP design.
    """

    frame_dim = 39

    def __init__(self, num_envs: int, device: torch.device | str, window_frames: int = 24, dt: float = 0.02):
        if int(window_frames) != 24:
            raise ValueError("Parallelism AMP currently requires window_frames=24")
        self.num_envs = int(num_envs)
        self.window_frames = int(window_frames)
        self.num_transitions = self.window_frames - 1
        self.dt = float(dt)
        self.device = torch.device(device)
        shape = (self.num_envs, self.num_transitions, self.frame_dim)
        self.agent_delta_ring = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.expert_delta_ring = torch.zeros_like(self.agent_delta_ring)
        self.write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.valid_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.agent_terminal = torch.zeros(self.num_envs, self.frame_dim, device=self.device)
        self.expert_terminal = torch.zeros_like(self.agent_terminal)
        self.last_agent_delta = torch.zeros_like(self.agent_terminal)
        self.last_expert_delta = torch.zeros_like(self.expert_terminal)
        self.standstill_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _batch(self, value: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        result = torch.as_tensor(value, device=self.device)
        if result.ndim != 2 or result.shape != (self.num_envs, self.frame_dim):
            raise ValueError(f"expected [{self.num_envs}, {self.frame_dim}], got {tuple(result.shape)}")
        return result.to(dtype=dtype or torch.float32)

    def reset(self, env_mask: Tensor | None = None) -> None:
        if env_mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            mask = torch.as_tensor(env_mask, device=self.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != self.num_envs:
                raise ValueError(f"expected env_mask with {self.num_envs} rows")
        self.agent_delta_ring[mask] = 0.0
        self.expert_delta_ring[mask] = 0.0
        self.write_index[mask] = 0
        self.valid_count[mask] = 0
        self.agent_terminal[mask] = 0.0
        self.expert_terminal[mask] = 0.0
        self.last_agent_delta[mask] = 0.0
        self.last_expert_delta[mask] = 0.0
        self.standstill_latched[mask] = False

    def _ordered_deltas(self, ring: Tensor) -> Tensor:
        offsets = torch.arange(self.num_transitions, device=self.device, dtype=torch.long)
        slots = (self.write_index[:, None] + offsets[None, :]) % self.num_transitions
        rows = torch.arange(self.num_envs, device=self.device)[:, None]
        return ring[rows, slots]

    def reconstruct_and_encode(self, terminal_state: Tensor, ordered_deltas: Tensor) -> Tensor:
        """Reconstruct and localize a 24-frame window from oldest-to-newest deltas."""

        terminal = self._batch(terminal_state)
        deltas = torch.as_tensor(ordered_deltas, device=self.device, dtype=torch.float32)
        expected = (self.num_envs, self.num_transitions, self.frame_dim)
        if tuple(deltas.shape) != expected:
            raise ValueError(f"expected {expected}, got {tuple(deltas.shape)}")
        reverse_sum = torch.flip(torch.cumsum(torch.flip(deltas, dims=(1,)), dim=1), dims=(1,))
        states = torch.empty((self.num_envs, self.window_frames, self.frame_dim), device=self.device)
        states[:, -1] = terminal
        states[:, :-1] = terminal[:, None, :] - reverse_sum

        # Root position is translated into the terminal-frame origin.  Heading
        # removal uses the terminal 6D rotation and preserves roll/pitch data.
        terminal_pos = states[:, -1, 24:27]
        states[:, :, 24:27] -= terminal_pos[:, None, :]
        rotations = _rotation_6d_to_matrix(states[:, :, 27:33])
        terminal_rot = rotations[:, -1]
        yaw = torch.atan2(terminal_rot[:, 1, 0], terminal_rot[:, 0, 0])
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        heading = torch.zeros((self.num_envs, 3, 3), device=self.device)
        heading[:, 0, 0], heading[:, 0, 1] = cy, -sy
        heading[:, 1, 0], heading[:, 1, 1] = sy, cy
        heading[:, 2, 2] = 1.0
        local_rot = torch.matmul(heading[:, None].transpose(-1, -2), rotations)
        states[:, :, 27:33] = _matrix_to_rotation_6d(local_rot)
        states[:, :, 33:36] = torch.matmul(
            heading[:, None].transpose(-1, -2), states[:, :, 33:36, None]
        ).squeeze(-1)
        states[:, :, 36:39] = torch.matmul(
            heading[:, None].transpose(-1, -2), states[:, :, 36:39, None]
        ).squeeze(-1)
        return states.reshape(self.num_envs, self.window_frames * self.frame_dim)

    def _build_payload(self, valid: Tensor) -> AmpStepPayload:
        agent = self.reconstruct_and_encode(self.agent_terminal, self._ordered_deltas(self.agent_delta_ring))
        expert = self.reconstruct_and_encode(self.expert_terminal, self._ordered_deltas(self.expert_delta_ring))
        active = valid & (self.valid_count == self.window_frames)
        ratio = self.valid_count.to(dtype=torch.float32) / float(self.window_frames)
        active_f = active[:, None, None]
        agent = torch.where(active_f.reshape(self.num_envs, 1, 1), agent.reshape(self.num_envs, self.window_frames, self.frame_dim), torch.zeros((self.num_envs, self.window_frames, self.frame_dim), device=self.device))
        expert = torch.where(active_f, expert.reshape(self.num_envs, self.window_frames, self.frame_dim), torch.zeros_like(agent))
        return AmpStepPayload(expert, agent, active, ratio)

    def compute_transition_deltas(
        self,
        agent_start: Tensor,
        agent_target: Tensor,
        expert_start: Tensor,
        expert_target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute all environment transition increments in one Torch op.

        The method deliberately has no per-environment Python work: callers
        provide ``[num_envs, 39]`` tensors and receive paired deltas with the
        same shape.  It is kept public so runtime probes can verify that the
        transition path is batched independently of ring-buffer maintenance.
        """

        agent_start = self._batch(agent_start)
        agent_target = self._batch(agent_target)
        expert_start = self._batch(expert_start)
        expert_target = self._batch(expert_target)
        return torch.sub(agent_target, agent_start), torch.sub(expert_target, expert_start)

    def push_transition(
        self,
        agent_start: Tensor,
        agent_target: Tensor,
        expert_start: Tensor,
        expert_target: Tensor,
        valid: Tensor,
    ) -> AmpStepPayload:
        """Push one batched transition and return the current paired windows."""

        valid = torch.as_tensor(valid, device=self.device, dtype=torch.bool).reshape(-1)
        if valid.numel() != self.num_envs:
            raise ValueError(f"expected valid with {self.num_envs} rows")
        invalid = ~valid
        self.agent_delta_ring[invalid] = 0.0
        self.expert_delta_ring[invalid] = 0.0
        self.valid_count[invalid] = 0
        self.write_index[invalid] = 0
        agent_delta, expert_delta = self.compute_transition_deltas(
            agent_start, agent_target, expert_start, expert_target
        )
        rows = torch.arange(self.num_envs, device=self.device)
        slots = self.write_index
        old_agent = self.agent_delta_ring[rows, slots]
        old_expert = self.expert_delta_ring[rows, slots]
        self.agent_delta_ring[rows, slots] = torch.where(valid[:, None], agent_delta, old_agent)
        self.expert_delta_ring[rows, slots] = torch.where(valid[:, None], expert_delta, old_expert)
        self.agent_terminal = torch.where(valid[:, None], agent_target, self.agent_terminal)
        self.expert_terminal = torch.where(valid[:, None], expert_target, self.expert_terminal)
        self.last_agent_delta = torch.where(valid[:, None], agent_delta, self.last_agent_delta)
        self.last_expert_delta = torch.where(valid[:, None], expert_delta, self.last_expert_delta)
        self.write_index = torch.where(
            valid, (self.write_index + 1) % self.num_transitions, torch.zeros_like(self.write_index)
        )
        # The first valid transition contains the frame anchor and target (2
        # frames); subsequent transitions add one frame each.
        increment = torch.where(self.valid_count == 0, torch.full_like(self.valid_count, 2), torch.ones_like(self.valid_count))
        self.valid_count = torch.where(
            valid, torch.clamp(self.valid_count + increment, max=self.window_frames), torch.zeros_like(self.valid_count)
        )
        self.standstill_latched = torch.where(invalid, torch.ones_like(self.standstill_latched), self.standstill_latched)
        self.standstill_latched = torch.where(valid, torch.zeros_like(self.standstill_latched), self.standstill_latched)
        return self._build_payload(valid)
