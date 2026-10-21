"""Low-cost M1 diagnostics for separating leg and wheel learning signals."""

from __future__ import annotations

import torch

from go2_pvcnn.assets.m1 import M1_USD_JOINT_NAMES
from m1_ame.m1_ame_rewards import M1_LEG_JOINT_NAMES, M1_WHEEL_JOINT_NAMES


def _rms(values: torch.Tensor, indices: list[int]) -> torch.Tensor | None:
    if not indices:
        return None
    selected = torch.as_tensor(values).index_select(
        -1, torch.as_tensor(indices, dtype=torch.long, device=values.device)
    )
    return torch.sqrt(torch.mean(torch.square(selected)))


def _joint_indices(robot):
    cache = getattr(robot, "_m1_ame_diagnostic_joint_indices", None)
    if cache is not None:
        return cache
    leg_ids = robot.find_joints(list(M1_LEG_JOINT_NAMES), preserve_order=True)[0]
    wheel_ids = robot.find_joints(list(M1_WHEEL_JOINT_NAMES), preserve_order=True)[0]
    action_index = {name: index for index, name in enumerate(M1_USD_JOINT_NAMES)}
    action_leg_ids = [action_index[name] for name in M1_LEG_JOINT_NAMES]
    action_wheel_ids = [action_index[name] for name in M1_WHEEL_JOINT_NAMES]
    cache = (leg_ids, wheel_ids, action_leg_ids, action_wheel_ids)
    robot._m1_ame_diagnostic_joint_indices = cache
    return cache


def collect_m1_joint_diagnostics(env) -> dict[str, torch.Tensor]:
    """Return scalar RMS diagnostics without adding work to the env step path."""

    root = getattr(env, "unwrapped", env)
    try:
        robot = root.scene["robot"]
        leg_ids, wheel_ids, action_leg_ids, action_wheel_ids = _joint_indices(robot)
    except (AttributeError, KeyError, TypeError, ValueError):
        return {}

    out: dict[str, torch.Tensor] = {}
    joint_vel = getattr(robot.data, "joint_vel", None)
    applied_torque = getattr(robot.data, "applied_torque", None)
    if joint_vel is not None:
        leg_velocity = _rms(joint_vel, leg_ids)
        wheel_velocity = _rms(joint_vel, wheel_ids)
        if leg_velocity is not None:
            out["leg_velocity_rms"] = torch.nan_to_num(leg_velocity)
            out["leg_velocity_abs_max"] = torch.nan_to_num(
                torch.as_tensor(joint_vel).index_select(
                    -1, torch.as_tensor(leg_ids, dtype=torch.long, device=joint_vel.device)
                ).abs().amax()
            )
        if wheel_velocity is not None:
            out["wheel_velocity_rms"] = torch.nan_to_num(wheel_velocity)
            out["wheel_velocity_abs_max"] = torch.nan_to_num(
                torch.as_tensor(joint_vel).index_select(
                    -1, torch.as_tensor(wheel_ids, dtype=torch.long, device=joint_vel.device)
                ).abs().amax()
            )
    if applied_torque is not None:
        leg_torque = _rms(applied_torque, leg_ids)
        wheel_torque = _rms(applied_torque, wheel_ids)
        if leg_torque is not None:
            out["leg_torque_rms"] = torch.nan_to_num(leg_torque)
        if wheel_torque is not None:
            out["wheel_torque_rms"] = torch.nan_to_num(wheel_torque)

    action_manager = getattr(root, "action_manager", None)
    action = getattr(action_manager, "action", None)
    previous_action = getattr(action_manager, "prev_action", None)
    if action is not None and previous_action is not None:
        action_rate = torch.as_tensor(action) - torch.as_tensor(previous_action)
        leg_action_rate = _rms(action_rate, action_leg_ids)
        wheel_action_rate = _rms(action_rate, action_wheel_ids)
        if leg_action_rate is not None:
            out["action_rate_leg_rms"] = torch.nan_to_num(leg_action_rate)
        if wheel_action_rate is not None:
            out["action_rate_wheel_rms"] = torch.nan_to_num(wheel_action_rate)
    return out


__all__ = ["collect_m1_joint_diagnostics"]
