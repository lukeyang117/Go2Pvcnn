"""Reference generator scaffolding for planner-guided training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core import HIP_HEIGHT, HIP_OFFSETS_ARRAY, LEG_ORDER

from .reference_cache import ReferenceTrajectoryCache


@dataclass
class ReferenceGeneratorConfig:
    """Configuration for future planner-backed reference generation."""

    horizon_steps: int = 50
    dt: float = 0.02
    forward_speed: float = 0.12


class ReferenceGenerator:
    """Build a tiny, import-safe placeholder reference trajectory."""

    def __init__(self, config: ReferenceGeneratorConfig | None = None):
        self.config = config or ReferenceGeneratorConfig()

    def generate(self) -> ReferenceTrajectoryCache:
        """Return a populated cache with a small forward drift."""
        horizon = int(self.config.horizon_steps)
        if horizon <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.config.dt <= 0:
            raise ValueError("dt must be positive")

        time = torch.arange(horizon, dtype=torch.float32) * float(self.config.dt)
        root_pos_w = torch.stack(
            (
                time * float(self.config.forward_speed),
                torch.zeros_like(time),
                torch.full_like(time, float(HIP_HEIGHT)),
            ),
            dim=-1,
        )

        root_quat_w = torch.zeros((horizon, 4), dtype=torch.float32)
        root_quat_w[:, 0] = 1.0

        joint_angles = torch.tensor(
            [
                0.1,
                0.7,
                -1.5,
                -0.1,
                0.7,
                -1.5,
                0.1,
                1.0,
                -1.5,
                -0.1,
                1.0,
                -1.5,
            ],
            dtype=torch.float32,
        ).unsqueeze(0).repeat(horizon, 1)

        foot_pos_root = torch.as_tensor(HIP_OFFSETS_ARRAY, dtype=torch.float32).clone()
        foot_pos_root[:, 2] = -float(HIP_HEIGHT)
        foot_pos_root = foot_pos_root.unsqueeze(0).repeat(horizon, 1, 1)

        contact_state = torch.ones((horizon, len(LEG_ORDER)), dtype=torch.bool)
        planned_touchdown_w = root_pos_w[:, None, :] + foot_pos_root
        phase_index = torch.arange(horizon, dtype=torch.long)
        valid_mask = torch.ones(horizon, dtype=torch.bool)

        return ReferenceTrajectoryCache(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            joint_angles=joint_angles,
            foot_pos_root=foot_pos_root,
            contact_state=contact_state,
            planned_touchdown_w=planned_touchdown_w,
            phase_index=phase_index,
            valid_mask=valid_mask,
        )
