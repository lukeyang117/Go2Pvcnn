"""Planner-independent fixed gait phase observations and rewards."""

from __future__ import annotations

import torch


_LEG_COUNT = 4
_DEFAULT_CYCLE_STEPS = 24
_DEFAULT_HALF_CYCLE_STEPS = 12


def _step_indices(env, *, offset: int = 0) -> torch.Tensor:
    steps = getattr(env, "episode_length_buf", None)
    if steps is None:
        steps = getattr(env, "common_step_counter", 0)
    if not isinstance(steps, torch.Tensor):
        steps = torch.full(
            (int(getattr(env, "num_envs", 1)),),
            int(steps),
            dtype=torch.long,
            device=getattr(env, "device", "cpu"),
        )
    steps = steps.to(device=getattr(env, "device", steps.device), dtype=torch.long).reshape(-1)
    return torch.clamp(steps + int(offset), min=0)


def fixed_gait_phase(env, cycle_steps: int = _DEFAULT_CYCLE_STEPS) -> torch.Tensor:
    """Return one normalized phase scalar per environment in ``[0, 1)``."""

    cycle = int(cycle_steps)
    if cycle <= 0:
        raise ValueError("cycle_steps must be positive")
    phase = torch.remainder(_step_indices(env), cycle).to(dtype=torch.float32)
    return (phase / float(cycle)).unsqueeze(-1)


def _target_stance_from_steps(
    steps: torch.Tensor,
    *,
    command: torch.Tensor,
    cycle_steps: int,
    half_cycle_steps: int,
    standstill_threshold: float,
) -> torch.Tensor:
    if cycle_steps <= 0 or half_cycle_steps <= 0 or half_cycle_steps * 2 != cycle_steps:
        raise ValueError("cycle_steps must be twice half_cycle_steps")
    first_half = torch.remainder(steps, cycle_steps) < half_cycle_steps
    stance = torch.ones(steps.shape[0], _LEG_COUNT, dtype=torch.bool, device=steps.device)
    # Parallelism's diagonal trot convention: FL/RR swing first, FR/RL second.
    stance[:, 0] = ~first_half
    stance[:, 3] = ~first_half
    stance[:, 1] = first_half
    stance[:, 2] = first_half
    command = torch.as_tensor(command, dtype=torch.float32, device=steps.device)
    moving = (torch.linalg.norm(command[:, :2], dim=-1) > float(standstill_threshold)) | (
        command[:, 2].abs() > float(standstill_threshold)
    )
    stance[~moving] = True
    return stance


def _command(env, command_name: str, *, device: torch.device) -> torch.Tensor:
    manager = getattr(env, "command_manager", None)
    if manager is None or not hasattr(manager, "get_command"):
        return torch.zeros(int(getattr(env, "num_envs", 1)), 3, dtype=torch.float32, device=device)
    command = manager.get_command(command_name)
    command = torch.as_tensor(command, dtype=torch.float32, device=device)
    if command.ndim != 2 or command.shape[-1] < 3:
        raise ValueError("velocity command must have shape [num_envs, 3 or more]")
    return command[:, :3]


def _contact_forces(env, sensor_cfg, *, device: torch.device) -> torch.Tensor:
    sensor_name = getattr(sensor_cfg, "name", "contact_forces")
    sensor = env.scene[sensor_name]
    forces = torch.as_tensor(sensor.data.net_forces_w, dtype=torch.float32, device=device)
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if body_ids is not None:
        ids = torch.as_tensor(body_ids, dtype=torch.long, device=device).reshape(-1)
        forces = forces.index_select(1, ids)
    if forces.ndim != 3 or forces.shape[1] != _LEG_COUNT or forces.shape[2] < 3:
        raise ValueError("foot contact forces must have shape [num_envs, 4, 3]")
    return forces[..., 2]


def fixed_gait_contact_match_reward(
    env,
    *,
    sensor_cfg=None,
    command_name: str = "base_velocity",
    cycle_steps: int = _DEFAULT_CYCLE_STEPS,
    half_cycle_steps: int = _DEFAULT_HALF_CYCLE_STEPS,
    standstill_threshold: float = 0.05,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward actual foot support/contact matching the fixed gait phase.

    Rewards are evaluated after a physics step, so ``episode_length_buf - 1``
    refers to the phase that produced the action being rewarded.
    """

    device = torch.device(getattr(env, "device", "cpu"))
    steps = _step_indices(env, offset=-1)
    command = _command(env, command_name, device=device)
    target_stance = _target_stance_from_steps(
        steps,
        command=command,
        cycle_steps=int(cycle_steps),
        half_cycle_steps=int(half_cycle_steps),
        standstill_threshold=float(standstill_threshold),
    )
    if sensor_cfg is None:
        class _SensorCfg:
            name = "contact_forces"
            body_ids = None

        sensor_cfg = _SensorCfg()
    actual_contact = _contact_forces(env, sensor_cfg, device=device) > float(contact_force_threshold)
    return (actual_contact == target_stance).to(dtype=torch.float32).mean(dim=-1)


__all__ = ["fixed_gait_phase", "fixed_gait_contact_match_reward"]
