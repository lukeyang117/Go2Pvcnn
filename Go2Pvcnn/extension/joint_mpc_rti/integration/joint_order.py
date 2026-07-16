"""Shared Isaac robot-order and joint-MPC planner-order conversions."""

from __future__ import annotations

import torch
from torch import Tensor


PLANNER_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


def _normalize_joint_name(name: str) -> str:
    normalized = str(name).split("/")[-1]
    normalized = normalized.split(":")[-1]
    return normalized.lower()


def joint_order_indices(*, source_order: tuple[str, ...], target_order: tuple[str, ...]) -> Tensor:
    source_to_index = {_normalize_joint_name(name): idx for idx, name in enumerate(source_order)}
    missing = [name for name in target_order if _normalize_joint_name(name) not in source_to_index]
    if missing:
        raise ValueError(f"joint order is missing required joints: {missing}")
    return torch.tensor(
        [source_to_index[_normalize_joint_name(name)] for name in target_order],
        dtype=torch.long,
    )


def reorder_joints(values: Tensor, *, source_order: tuple[str, ...], target_order: tuple[str, ...]) -> Tensor:
    tensor = torch.as_tensor(values)
    if int(tensor.shape[-1]) != len(source_order):
        raise ValueError("joint tensor width must match source_order")
    indices = joint_order_indices(source_order=source_order, target_order=target_order)
    return tensor.index_select(-1, indices.to(device=tensor.device))


def robot_to_planner_joints(values: Tensor, robot_joint_names) -> Tensor:
    if not robot_joint_names:
        return torch.as_tensor(values)
    return reorder_joints(
        values,
        source_order=tuple(robot_joint_names),
        target_order=PLANNER_JOINT_ORDER,
    )


def planner_to_robot_joints(values: Tensor, robot_joint_names) -> Tensor:
    if not robot_joint_names:
        return torch.as_tensor(values)
    return reorder_joints(
        values,
        source_order=PLANNER_JOINT_ORDER,
        target_order=tuple(robot_joint_names),
    )


__all__ = [
    "PLANNER_JOINT_ORDER",
    "joint_order_indices",
    "planner_to_robot_joints",
    "reorder_joints",
    "robot_to_planner_joints",
]
