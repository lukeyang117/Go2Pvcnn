from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CommandCase:
    name: str
    command: tuple[float, float, float]


COMMAND_CASES: tuple[CommandCase, ...] = (
    CommandCase("standstill", (0.0, 0.0, 0.0)),
    CommandCase("forward", (0.6, 0.0, 0.0)),
    CommandCase("backward", (-0.6, 0.0, 0.0)),
    CommandCase("lateral_left", (0.0, 0.35, 0.0)),
    CommandCase("lateral_right", (0.0, -0.35, 0.0)),
    CommandCase("yaw_left", (0.0, 0.0, 0.5)),
    CommandCase("yaw_right", (0.0, 0.0, -0.5)),
)


def build_command_cases(*, device: torch.device, num_envs: int) -> dict[str, torch.Tensor]:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")

    return {
        case.name: torch.tensor(case.command, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1).clone()
        for case in COMMAND_CASES
    }
