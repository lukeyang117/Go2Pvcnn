"""24-frame planner tracking alignment with explicit validity boundaries."""

from __future__ import annotations

import torch
from torch import Tensor


class ValidWindowBuffer:
    """Collect complete windows without crossing reset or replan boundaries."""

    def __init__(self, num_envs: int, horizon: int = 24, device: torch.device | str = "cpu") -> None:
        if int(horizon) != 24:
            raise ValueError("horizon must be 24")
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.device = torch.device(device)
        self.actual = torch.zeros(self.num_envs, self.horizon, 39, device=self.device)
        self.reference = torch.zeros_like(self.actual)
        self.plan_ids = torch.full((self.num_envs, self.horizon), -1, dtype=torch.long, device=self.device)
        self.count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.valid_transitions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_plan_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

    def reset(self, mask: Tensor | None = None) -> None:
        rows = torch.ones(self.num_envs, dtype=torch.bool, device=self.device) if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
        self.actual[rows] = 0.0
        self.reference[rows] = 0.0
        self.plan_ids[rows] = -1
        self.count[rows] = 0
        self.valid_transitions[rows] = 0
        self._last_plan_id[rows] = -1

    def push(self, actual_frame: Tensor, reference_frame: Tensor, plan_id: Tensor, plan_valid: Tensor, reset: Tensor, transition_valid: Tensor) -> Tensor:
        actual = torch.as_tensor(actual_frame, dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(reference_frame, dtype=torch.float32, device=self.device)
        plan_id = torch.as_tensor(plan_id, dtype=torch.long, device=self.device).reshape(-1)
        plan_valid = torch.as_tensor(plan_valid, dtype=torch.bool, device=self.device).reshape(-1)
        reset = torch.as_tensor(reset, dtype=torch.bool, device=self.device).reshape(-1)
        transition_valid = torch.as_tensor(transition_valid, dtype=torch.bool, device=self.device).reshape(-1)
        if actual.shape != (self.num_envs, 39) or reference.shape != actual.shape:
            raise ValueError(f"frames must have shape [{self.num_envs}, 39]")
        if any(value.numel() != self.num_envs for value in (plan_id, plan_valid, reset, transition_valid)):
            raise ValueError("window metadata must match num_envs")
        changed = (self._last_plan_id >= 0) & (plan_id != self._last_plan_id)
        invalid = reset | ~plan_valid | ~transition_valid | changed
        self.reset(invalid)
        accept = ~invalid
        rows = torch.arange(self.num_envs, device=self.device)
        slots = torch.clamp(self.count, max=self.horizon - 1)
        self.actual[rows, slots] = torch.where(accept[:, None], actual, self.actual[rows, slots])
        self.reference[rows, slots] = torch.where(accept[:, None], reference, self.reference[rows, slots])
        self.plan_ids[rows, slots] = torch.where(accept, plan_id, self.plan_ids[rows, slots])
        first = accept & (self.count == 0)
        # A transition contributes B0 and B1 on first use, then one frame per transition.
        self.count = torch.where(accept, torch.clamp(self.count + torch.where(first, 2, 1), max=self.horizon), self.count)
        self.valid_transitions = torch.where(accept, torch.clamp(self.valid_transitions + 1, max=self.horizon - 1), self.valid_transitions)
        self._last_plan_id = torch.where(accept, plan_id, self._last_plan_id)
        return accept & (self.count == self.horizon) & (self.valid_transitions == self.horizon - 1)


def tracking_mse(actual: Tensor, reference: Tensor, valid_mask: Tensor, anchor_root: Tensor) -> dict[str, Tensor]:
    actual = torch.as_tensor(actual, dtype=torch.float32)
    reference = torch.as_tensor(reference, dtype=torch.float32, device=actual.device)
    valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=actual.device)
    if actual.ndim != 3 or actual.shape[-1] != 39 or reference.shape != actual.shape:
        raise ValueError("actual and reference must have shape [N, H, 39]")
    if valid.shape != actual.shape[:2]:
        raise ValueError("valid_mask must have shape [N, H]")
    weights = valid.to(actual.dtype)
    denom = weights.sum().clamp_min(1.0)
    joint_error = (actual[..., :12] - reference[..., :12]).square().mean(dim=-1)
    joint_vel_error = (actual[..., 12:24] - reference[..., 12:24]).square().mean(dim=-1)
    actual_root = actual[..., 24:27] - actual[:, :1, 24:27]
    reference_root = reference[..., 24:27] - reference[:, :1, 24:27]
    root_error = (actual_root - reference_root).square().mean(dim=-1)
    rotation_error = (actual[..., 27:33] - reference[..., 27:33]).square().mean(dim=-1)
    lin_vel_error = (actual[..., 33:36] - reference[..., 33:36]).square().mean(dim=-1)
    ang_vel_error = (actual[..., 36:39] - reference[..., 36:39]).square().mean(dim=-1)
    _ = anchor_root  # Kept in the public API to make the coordinate anchor explicit.
    reduce = lambda value: (value * weights).sum() / denom
    return {
        "joint_position_mse": reduce(joint_error),
        "joint_velocity_mse": reduce(joint_vel_error),
        "root_position_mse": reduce(root_error),
        "root_rotation_mse": reduce(rotation_error),
        "root_linear_velocity_mse": reduce(lin_vel_error),
        "root_angular_velocity_mse": reduce(ang_vel_error),
        "valid_window_ratio": valid.all(dim=1).to(torch.float32).mean(),
    }
