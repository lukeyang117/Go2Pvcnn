"""Minimal trajectory generation for the Go2 planner core.

This module intentionally only guarantees a standstill reference path for now.
It stays import-safe and pure NumPy so runtime wiring can grow on top later.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import TrajectoryConfig
from .ik import (
    batch_body_pos_root_relative,
    batch_forward_kinematics,
    batch_inverse_kinematics,
)
from .types import (
    Command,
    HIP_HEIGHT,
    HIP_OFFSETS_ARRAY,
    RobotState,
    TrajectoryResult,
    quat_from_yaw,
    rotate_vector,
)

_STANDSTILL_CMD_EPS = 1e-5


class TerrainLike(Protocol):
    """Small terrain interface accepted by the minimal trajectory path."""

    def height_at(self, x: float, y: float) -> float: ...


class _FlatTerrain:
    """Flat z=0 terrain used when no terrain object is provided."""

    def height_at(self, x: float, y: float) -> float:
        return 0.0


def command_is_standstill(cmd: Command, eps: float = _STANDSTILL_CMD_EPS) -> bool:
    """Return True when the command is effectively zero."""
    return abs(cmd.vx) <= eps and abs(cmd.vy) <= eps and abs(cmd.yaw_rate) <= eps


def _terrain_height(terrain: TerrainLike | None, x: float, y: float) -> float:
    if terrain is None:
        return 0.0
    return float(terrain.height_at(float(x), float(y)))


def default_initial_state(
    terrain: TerrainLike | None = None,
    x: float = 0.0,
    y: float = 0.0,
) -> RobotState:
    """Create a default standing state for Go2 at the given xy position."""
    ground_z = _terrain_height(terrain, x, y)
    root_pos = np.array([x, y, ground_z + HIP_HEIGHT], dtype=np.float64)
    root_quat = quat_from_yaw(0.0)

    hip_positions = root_pos + rotate_vector(HIP_OFFSETS_ARRAY, root_quat)
    foot_pos = np.empty((4, 3), dtype=np.float64)
    for leg_idx in range(4):
        foot_pos[leg_idx, 0] = hip_positions[leg_idx, 0]
        foot_pos[leg_idx, 1] = hip_positions[leg_idx, 1]
        foot_pos[leg_idx, 2] = _terrain_height(
            terrain,
            float(foot_pos[leg_idx, 0]),
            float(foot_pos[leg_idx, 1]),
        )

    joint_angles = batch_inverse_kinematics(
        root_pos.reshape(1, 3),
        root_quat.reshape(1, 4),
        foot_pos.reshape(1, 4, 3),
    )[0]

    return RobotState(
        root_pos=root_pos,
        root_quat=root_quat,
        joint_angles=joint_angles,
        foot_pos=foot_pos,
    )


def generate_standstill_trajectory(
    initial_state: RobotState,
    n_frames: int,
    dt: float,
) -> TrajectoryResult:
    """Generate a reference trajectory with the robot holding a fixed stance."""
    del dt
    root_pos = np.tile(np.asarray(initial_state.root_pos, dtype=np.float64), (n_frames, 1))
    root_quat = np.tile(np.asarray(initial_state.root_quat, dtype=np.float64), (n_frames, 1))
    foot_targets = np.tile(
        np.asarray(initial_state.foot_pos, dtype=np.float64).reshape(1, 4, 3),
        (n_frames, 1, 1),
    )
    contact_seq = np.ones((n_frames, 4), dtype=np.float32)
    joint_angles = batch_inverse_kinematics(root_pos, root_quat, foot_targets)
    body_pos_w = batch_forward_kinematics(root_pos, root_quat, joint_angles)
    body_pos_root = batch_body_pos_root_relative(root_pos, root_quat, body_pos_w)
    foot_pos_root = batch_body_pos_root_relative(root_pos, root_quat, foot_targets)
    root_lin_vel = np.zeros((n_frames, 3), dtype=np.float64)
    root_ang_vel = np.zeros((n_frames, 3), dtype=np.float64)
    planned_td = np.asarray(initial_state.foot_pos, dtype=np.float64).reshape(4, 3)

    return TrajectoryResult(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        root_lin_vel_w=root_lin_vel,
        root_ang_vel_w=root_ang_vel,
        joint_angles=joint_angles,
        foot_pos_w=foot_targets,
        foot_pos_root=foot_pos_root,
        contact_state=contact_seq,
        body_pos_root=body_pos_root,
        planned_touchdown_w=planned_td,
    )


def generate_trajectory(
    terrain: TerrainLike | None,
    initial_state: RobotState,
    command: Command,
    n_frames: int,
    dt: float = 0.02,
    config: TrajectoryConfig | None = None,
) -> TrajectoryResult:
    """Generate the minimal reference trajectory path.

    Only standstill is implemented in this core slice. Non-zero commands raise
    ``NotImplementedError`` so the missing path stays explicit.
    """
    del terrain, config
    if not command_is_standstill(command):
        raise NotImplementedError(
            "non-standstill trajectory generation is not ported into core yet"
        )
    return generate_standstill_trajectory(initial_state, n_frames, dt)


__all__ = [
    "Command",
    "RobotState",
    "TerrainLike",
    "TrajectoryResult",
    "command_is_standstill",
    "default_initial_state",
    "generate_standstill_trajectory",
    "generate_trajectory",
]
