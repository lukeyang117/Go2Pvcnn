"""GPU-batch episode metrics used by the benchmark rollout."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class EpisodeAccumulator:
    """Keep sticky episode metrics for all environments without row loops."""

    num_envs: int
    device: torch.device | str
    _previous_root: Tensor = field(init=False)
    _initialized: Tensor = field(init=False)
    large_collision: Tensor = field(init=False)
    small_collision: Tensor = field(init=False)
    fallen: Tensor = field(init=False)
    path_m: Tensor = field(init=False)
    progress_m: Tensor = field(init=False)
    contact_steps: Tensor = field(init=False)
    valid_steps: Tensor = field(init=False)
    total_steps: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.num_envs = int(self.num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.device = torch.device(self.device)
        shape = (self.num_envs,)
        self._previous_root = torch.zeros(self.num_envs, 3, device=self.device)
        self._initialized = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.large_collision = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.small_collision = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.fallen = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.path_m = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.progress_m = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.contact_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.valid_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.total_steps = torch.zeros(shape, dtype=torch.long, device=self.device)

    def update(
        self,
        root_pos_w: Tensor,
        command: Tensor,
        large_contact: Tensor,
        small_contact: Tensor,
        fallen: Tensor,
        done: Tensor,
        valid_step: Tensor,
    ) -> None:
        root = torch.as_tensor(root_pos_w, dtype=torch.float32, device=self.device)
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if root.shape != (self.num_envs, 3) or command.shape != (self.num_envs, 3):
            raise ValueError("root_pos_w and command must have shape [num_envs, 3]")
        large = torch.as_tensor(large_contact, dtype=torch.bool, device=self.device).reshape(-1)
        small = torch.as_tensor(small_contact, dtype=torch.bool, device=self.device).reshape(-1)
        fall = torch.as_tensor(fallen, dtype=torch.bool, device=self.device).reshape(-1)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device).reshape(-1)
        valid = torch.as_tensor(valid_step, dtype=torch.bool, device=self.device).reshape(-1)
        if any(value.numel() != self.num_envs for value in (large, small, fall, done, valid)):
            raise ValueError("per-environment metric masks must match num_envs")

        delta = root - self._previous_root
        active = self._initialized
        distance = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        command_xy = command[..., :2]
        command_norm = torch.linalg.vector_norm(command_xy, dim=-1).clamp_min(1.0e-6)
        progress = (delta[..., :2] * command_xy).sum(dim=-1) / command_norm
        self.path_m += torch.where(active, distance, torch.zeros_like(distance))
        self.progress_m += torch.where(active, progress, torch.zeros_like(progress))
        self._previous_root.copy_(root)
        self._initialized[:] = True
        self.large_collision |= large
        self.small_collision |= small
        self.fallen |= fall
        self.contact_steps += (large | small).to(torch.long)
        self.valid_steps += valid.to(torch.long)
        self.total_steps += torch.ones_like(self.total_steps)

    def finish(self, env_ids: Tensor | None = None) -> list[dict[str, object]]:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        records: list[dict[str, object]] = []
        for env_id in ids.detach().cpu().tolist():
            index = int(env_id)
            records.append(
                {
                    "env_id": index,
                    "large_collision_episode": bool(self.large_collision[index].item()),
                    "small_collision_episode": bool(self.small_collision[index].item()),
                    "fall": bool(self.fallen[index].item()),
                    "traversed_path_m": float(self.path_m[index].item()),
                    "command_progress_m": float(self.progress_m[index].item()),
                    "contact_step_rate": float(self.contact_steps[index].item() / max(1, self.total_steps[index].item())),
                    "valid_step_count": int(self.valid_steps[index].item()),
                    "step_count": int(self.total_steps[index].item()),
                }
            )
        self._reset_rows(ids)
        return records

    def _reset_rows(self, ids: Tensor) -> None:
        self._previous_root[ids] = 0.0
        self._initialized[ids] = False
        self.large_collision[ids] = False
        self.small_collision[ids] = False
        self.fallen[ids] = False
        self.path_m[ids] = 0.0
        self.progress_m[ids] = 0.0
        self.contact_steps[ids] = 0
        self.valid_steps[ids] = 0
        self.total_steps[ids] = 0
