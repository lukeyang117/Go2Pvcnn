"""IsaacLab-style environment tensor extraction for joint MPC RTI."""

from __future__ import annotations

import math

import torch

from extension.convention import extract_roll_pitch_batch, extract_yaw_batch
from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
from extension.joint_mpc_rti.types import JointMpcRtiState


def _env_root(env):
    return getattr(env, "unwrapped", env)


def _named(container, name: str):
    if isinstance(container, dict):
        return container[name]
    try:
        return container[name]
    except (KeyError, TypeError):
        return getattr(container, name)


def state_from_env(env, *, device) -> JointMpcRtiState:
    root = _env_root(env)
    robot = _named(root.scene, "robot")
    data = robot.data
    root_quat = torch.as_tensor(data.root_quat_w, dtype=torch.float32, device=device)
    roll, pitch = extract_roll_pitch_batch(root_quat)
    yaw = extract_yaw_batch(root_quat)
    root_pos = torch.as_tensor(data.root_pos_w, dtype=torch.float32, device=device)
    joint_pos = torch.as_tensor(data.joint_pos, dtype=torch.float32, device=device)
    return JointMpcRtiState(
        root_pos_w=root_pos,
        root_rpy_w=torch.stack((roll, pitch, yaw), dim=-1),
        joint_pos=joint_pos,
        root_lin_vel_b=torch.as_tensor(
            getattr(data, "root_lin_vel_b", torch.zeros_like(root_pos)), dtype=torch.float32, device=device
        ),
        root_ang_vel_b=torch.as_tensor(
            getattr(data, "root_ang_vel_b", torch.zeros_like(root_pos)), dtype=torch.float32, device=device
        ),
        joint_vel=torch.as_tensor(
            getattr(data, "joint_vel", torch.zeros_like(joint_pos)), dtype=torch.float32, device=device
        ),
    )


def command_from_env(env, *, device, command_name: str = "base_velocity") -> torch.Tensor:
    root = _env_root(env)
    command = root.command_manager.get_command(command_name)
    tensor = torch.as_tensor(command, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or int(tensor.shape[-1]) < 3:
        raise ValueError("Isaac command must have shape [B,3 or more]")
    return tensor[:, :3].contiguous()


def field_from_env(
    env,
    *,
    device,
    version: torch.Tensor,
    scanner_name: str = "semantic_height_scanner",
    resolution: float = 0.01,
    small_ids: tuple[int, ...] = (1,),
    large_ids: tuple[int, ...] = (2,),
):
    root = _env_root(env)
    scene = root.scene
    scanner = _named(scene, scanner_name)
    data = scanner.data
    ray_hits = torch.as_tensor(data.ray_hits_w, dtype=torch.float32, device=device)
    if ray_hits.ndim != 3 or int(ray_hits.shape[-1]) != 3:
        raise ValueError("scanner ray_hits_w must have shape [B,N,3]")
    side = int(round(math.sqrt(int(ray_hits.shape[1]))))
    if side * side != int(ray_hits.shape[1]):
        raise ValueError("scanner ray count must form a square grid")
    height = ray_hits[..., 2].reshape(ray_hits.shape[0], side, side)
    semantic_value = getattr(data, "semantic_map", None)
    semantic = (
        torch.zeros_like(height, dtype=torch.long)
        if semantic_value is None
        else torch.as_tensor(semantic_value, dtype=torch.long, device=device).reshape(ray_hits.shape[0], side, side)
    )
    sensor_pos = torch.as_tensor(data.pos_w, dtype=torch.float32, device=device)
    sensor_quat = torch.as_tensor(data.quat_w, dtype=torch.float32, device=device)
    timestamp_value = getattr(data, "timestamp", None)
    timestamp = (
        version.to(dtype=torch.float32) * 0.02
        if timestamp_value is None
        else torch.as_tensor(timestamp_value, dtype=torch.float32, device=device).reshape(-1)
    )
    return build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=sensor_pos,
        yaw_w=extract_yaw_batch(sensor_quat),
        timestamp=timestamp,
        version=version,
        resolution=resolution,
        small_ids=small_ids,
        large_ids=large_ids,
    )


__all__ = ["command_from_env", "field_from_env", "state_from_env"]
