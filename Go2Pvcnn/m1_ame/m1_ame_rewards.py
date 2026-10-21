"""M1-sized rewards and live geometry checks for pure PPO AME."""

from __future__ import annotations

import torch
from torch import Tensor

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import go2_pvcnn.mdp as go2_mdp
from extension.convention import extract_roll_pitch_batch, extract_yaw_batch
from extension.parallelism.collision import official_collision_mask
from extension.parallelism.m1_kinematics import (
    M1_ASSET_JOINT_NAMES,
    M1_CFG as M1_PARALLEL_CFG,
    M1_PLANNER_JOINT_NAMES,
    M1_ROOT_Z_M,
    M1_WHEEL_HORIZONTAL_ENVELOPE_M,
    M1_WHEEL_RADIUS_M,
    M1_WHEEL_THICKNESS_M,
    m1_fk,
)
from extension.parallelism.robot_backend import RobotBackend
from extension.parallelism.types import ParallelismTerrain
from go2_pvcnn.assets.m1 import M1_USD_JOINT_NAMES
from tracking.cross_large_complex_ppo_env_cfg import CrossLargeComplexPpoRewardsCfg
from tracking.m1_parallelism_viewer_env_cfg import M1_SUPPORT_BODY_NAMES
from tracking import mdp as tracking_mdp
from tracking.mdp.policy_geometry_rewards import parallelism_terrain_from_scan
from m1_ame.m1_ame_safety import (
    finite_safe_quaternion_wxyz,
    finite_safe_tensor,
    m1_finite_reward,
    m1_invalid_state_mask,
)


M1_LEG_JOINT_NAMES = tuple(M1_PLANNER_JOINT_NAMES)
M1_ALL_JOINT_NAMES = tuple(M1_USD_JOINT_NAMES)
M1_WHEEL_JOINT_NAMES = tuple(name for name in M1_ALL_JOINT_NAMES if name.endswith("FOOT_JOINT"))
M1_LEG_LINK_NAMES = tuple(
    f"{prefix}_{part}_LINK"
    for prefix in ("FBL", "FAR", "RBL", "RAR")
    for part in ("ABAD", "HIP", "KNEE")
)
M1_UNDESIRED_CONTACT_BODY_NAMES = M1_LEG_LINK_NAMES
M1_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=list(M1_LEG_JOINT_NAMES))
M1_ALL_JOINT_CFG = SceneEntityCfg("robot", joint_names=list(M1_ALL_JOINT_NAMES))
M1_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=list(M1_WHEEL_JOINT_NAMES))
M1_WHEEL_BODY_CFG = SceneEntityCfg("robot", body_names=list(M1_SUPPORT_BODY_NAMES))
M1_SUPPORT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=list(M1_SUPPORT_BODY_NAMES))
M1_SUPPORT_ASSET_CFG = SceneEntityCfg("robot", body_names=list(M1_SUPPORT_BODY_NAMES))

_M1_PLANNER_INDICES = tuple(M1_ASSET_JOINT_NAMES.index(name) for name in M1_PLANNER_JOINT_NAMES)


def _joint_ids(asset, asset_cfg: SceneEntityCfg, names: tuple[str, ...]) -> list[int]:
    ids = getattr(asset_cfg, "joint_ids", None)
    if ids is None or isinstance(ids, slice):
        ids = asset.find_joints(list(names), preserve_order=True)[0]
    return list(ids)


def _safe_nonnegative(value: Tensor, upper: float = 1.0e8) -> Tensor:
    return finite_safe_tensor(torch.as_tensor(value), lower=0.0, upper=upper)


@m1_finite_reward
def m1_joint_vel_l2(env, asset_cfg: SceneEntityCfg) -> Tensor:
    asset = env.scene[asset_cfg.name]
    ids = _joint_ids(asset, asset_cfg, M1_LEG_JOINT_NAMES)
    velocity = finite_safe_tensor(asset.data.joint_vel[:, ids], lower=-20.0, upper=20.0)
    return _safe_nonnegative(torch.sum(torch.square(velocity), dim=1))


@m1_finite_reward
def m1_joint_acc_l2(env, asset_cfg: SceneEntityCfg) -> Tensor:
    asset = env.scene[asset_cfg.name]
    ids = _joint_ids(asset, asset_cfg, M1_ALL_JOINT_NAMES)
    acceleration = finite_safe_tensor(asset.data.joint_acc[:, ids], lower=-1.0e5, upper=1.0e5)
    return _safe_nonnegative(torch.sum(torch.square(acceleration), dim=1))


@m1_finite_reward
def m1_joint_torques_l2(env, asset_cfg: SceneEntityCfg) -> Tensor:
    asset = env.scene[asset_cfg.name]
    ids = _joint_ids(asset, asset_cfg, M1_ALL_JOINT_NAMES)
    torque = finite_safe_tensor(asset.data.applied_torque[:, ids], lower=-150.0, upper=150.0)
    return _safe_nonnegative(torch.sum(torch.square(torque), dim=1))


@m1_finite_reward
def m1_joint_pos_limits(env, asset_cfg: SceneEntityCfg) -> Tensor:
    asset = env.scene[asset_cfg.name]
    ids = _joint_ids(asset, asset_cfg, M1_LEG_JOINT_NAMES)
    position = finite_safe_tensor(asset.data.joint_pos[:, ids], lower=-4.0, upper=4.0)
    limits = finite_safe_tensor(asset.data.soft_joint_pos_limits[:, ids], lower=-4.0, upper=4.0)
    lower = limits[..., 0]
    upper = limits[..., 1]
    violation = torch.clamp(lower - position, min=0.0) + torch.clamp(position - upper, min=0.0)
    return _safe_nonnegative(torch.sum(violation, dim=1))


@m1_finite_reward
def m1_energy(env, asset_cfg: SceneEntityCfg) -> Tensor:
    asset = env.scene[asset_cfg.name]
    ids = _joint_ids(asset, asset_cfg, M1_ALL_JOINT_NAMES)
    qvel = finite_safe_tensor(asset.data.joint_vel[:, ids], lower=-25.0, upper=25.0)
    torque = finite_safe_tensor(asset.data.applied_torque[:, ids], lower=-150.0, upper=150.0)
    return _safe_nonnegative(torch.sum(torch.abs(qvel) * torch.abs(torque), dim=-1))


@m1_finite_reward
def m1_joint_position_penalty(
    env, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> Tensor:
    asset = env.scene[asset_cfg.name]
    ids = _joint_ids(asset, asset_cfg, M1_LEG_JOINT_NAMES)
    joint_pos = finite_safe_tensor(asset.data.joint_pos[:, ids], lower=-4.0, upper=4.0)
    default_pos = finite_safe_tensor(asset.data.default_joint_pos[:, ids], lower=-4.0, upper=4.0)
    command = finite_safe_tensor(env.command_manager.get_command("base_velocity"), lower=-5.0, upper=5.0)
    body_velocity = finite_safe_tensor(asset.data.root_lin_vel_b[:, :2], lower=-25.0, upper=25.0)
    error = torch.linalg.vector_norm(joint_pos - default_pos, dim=1)
    scale = torch.where(
        torch.logical_or(torch.linalg.vector_norm(command, dim=1) > 0.0, torch.linalg.vector_norm(body_velocity, dim=1) > velocity_threshold),
        error,
        float(stand_still_scale) * error,
    )
    return _safe_nonnegative(scale)


@m1_finite_reward
def m1_feet_air_time(env, command_name, sensor_cfg, threshold) -> Tensor:
    return finite_safe_tensor(
        go2_mdp.feet_air_time(env, command_name, sensor_cfg, threshold), lower=-100.0, upper=100.0
    )


@m1_finite_reward
def m1_air_time_variance(env, sensor_cfg) -> Tensor:
    return _safe_nonnegative(go2_mdp.air_time_variance_penalty(env, sensor_cfg), upper=100.0)


@m1_finite_reward
def m1_feet_slide(env, sensor_cfg, asset_cfg) -> Tensor:
    return _safe_nonnegative(go2_mdp.feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg), upper=100.0)


@m1_finite_reward
def m1_track_lin_vel_xy_exp(env, std, command_name, asset_cfg=SceneEntityCfg("robot")) -> Tensor:
    return _safe_nonnegative(go2_mdp.track_lin_vel_xy_exp(env, std, command_name, asset_cfg), upper=1.0)


@m1_finite_reward
def m1_track_ang_vel_z_exp(env, std, command_name, asset_cfg=SceneEntityCfg("robot")) -> Tensor:
    return _safe_nonnegative(go2_mdp.track_ang_vel_z_exp(env, std, command_name, asset_cfg), upper=1.0)


@m1_finite_reward
def m1_base_linear_velocity(env, asset_cfg=SceneEntityCfg("robot")) -> Tensor:
    return _safe_nonnegative(isaac_mdp.lin_vel_z_l2(env, asset_cfg=asset_cfg))


@m1_finite_reward
def m1_base_angular_velocity(env, asset_cfg=SceneEntityCfg("robot")) -> Tensor:
    return _safe_nonnegative(isaac_mdp.ang_vel_xy_l2(env, asset_cfg=asset_cfg))


@m1_finite_reward
def m1_flat_orientation(env, asset_cfg=SceneEntityCfg("robot")) -> Tensor:
    return _safe_nonnegative(go2_mdp.flat_orientation_l2(env, asset_cfg=asset_cfg))


@m1_finite_reward
def m1_action_rate(env) -> Tensor:
    return _safe_nonnegative(go2_mdp.action_rate_l2(env))


@m1_finite_reward
def m1_undesired_contacts(env, threshold, sensor_cfg) -> Tensor:
    return _safe_nonnegative(isaac_mdp.undesired_contacts(env, threshold=threshold, sensor_cfg=sensor_cfg), upper=100.0)


@m1_finite_reward
def m1_fixed_gait_contact_match(
    env,
    sensor_cfg=None,
    command_name="base_velocity",
    cycle_steps=24,
    half_cycle_steps=12,
    standstill_threshold=0.05,
    contact_force_threshold=1.0,
) -> Tensor:
    return _safe_nonnegative(
        tracking_mdp.fixed_gait_contact_match_reward(
            env,
            sensor_cfg=sensor_cfg,
            command_name=command_name,
            cycle_steps=cycle_steps,
            half_cycle_steps=half_cycle_steps,
            standstill_threshold=standstill_threshold,
            contact_force_threshold=contact_force_threshold,
        ),
        upper=1.0,
    )


def m1_termination_invalid_state(
    env,
    asset_cfg: SceneEntityCfg,
    leg_joint_cfg: SceneEntityCfg,
    wheel_joint_cfg: SceneEntityCfg,
) -> Tensor:
    robot = env.scene[asset_cfg.name]
    leg_ids = _joint_ids(robot, leg_joint_cfg, M1_LEG_JOINT_NAMES)
    wheel_ids = _joint_ids(robot, wheel_joint_cfg, M1_WHEEL_JOINT_NAMES)
    invalid = m1_invalid_state_mask(
        robot.data,
        leg_joint_ids=leg_ids,
        wheel_joint_ids=wheel_ids,
        leg_velocity_limit=20.0,
        wheel_velocity_limit=8.0,
    )
    env._m1_invalid_state_last = invalid
    return invalid


def _m1_geometry_collision_event(
    root_pos_w: Tensor,
    root_quat_w: Tensor,
    joint_pos: Tensor,
    terrain: ParallelismTerrain,
) -> Tensor:
    root_pos = torch.as_tensor(root_pos_w, dtype=torch.float32)
    root_quat = finite_safe_quaternion_wxyz(
        torch.as_tensor(root_quat_w, dtype=root_pos.dtype, device=root_pos.device)
    )
    indices = torch.as_tensor(_M1_PLANNER_INDICES, dtype=torch.long, device=root_pos.device)
    planner_joint = finite_safe_tensor(
        torch.as_tensor(joint_pos, dtype=root_pos.dtype, device=root_pos.device), lower=-4.0, upper=4.0
    ).index_select(-1, indices)
    roll, pitch = extract_roll_pitch_batch(root_quat)
    yaw = extract_yaw_batch(root_quat)
    root_rpy = torch.stack((roll, pitch, yaw), dim=-1)
    geometry = m1_fk(
        root_pos,
        root_rpy,
        planner_joint,
        capsule_samples=int(M1_PARALLEL_CFG.capsule_samples),
    )

    def pose(value: Tensor) -> Tensor:
        return value[:, None, None].expand(
            value.shape[0],
            value.shape[1],
            1,
            *value.shape[1:],
        )

    def samples(value: Tensor) -> Tensor:
        return value[:, None, None].expand(
            value.shape[0],
            value.shape[1],
            1,
            *value.shape[1:],
        )

    expanded = type(geometry)(
        hip_pos_w=pose(geometry.hip_pos_w),
        hip_rot_w=pose(geometry.hip_rot_w),
        foot_pos_w=pose(geometry.foot_pos_w),
        knee_pos_w=pose(geometry.knee_pos_w),
        calf_samples_w=samples(geometry.calf_samples_w),
        thigh_samples_w=samples(geometry.thigh_samples_w),
        thigh_pos_w=pose(geometry.thigh_pos_w),
        thigh_rot_w=pose(geometry.thigh_rot_w),
        calf_pos_w=pose(geometry.calf_pos_w),
        calf_rot_w=pose(geometry.calf_rot_w),
        foot_rot_w=pose(geometry.foot_rot_w),
    )
    _, collision_bits = official_collision_mask(
        terrain,
        expanded,
        M1_PARALLEL_CFG,
    )
    return collision_bits.any(dim=(1, 2, 3)).to(dtype=root_pos.dtype)


@m1_finite_reward
def m1_geometry_collision_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scanner_cfg: SceneEntityCfg = SceneEntityCfg("semantic_height_scanner"),
) -> Tensor:
    """Return the M1 live collision event using M1 USD geometry."""

    robot = env.scene[asset_cfg.name]
    scanner = env.scene[scanner_cfg.name]
    root_pos = torch.as_tensor(robot.data.root_pos_w, dtype=torch.float32)
    scanner_data = scanner.data
    ray_hits = getattr(scanner_data, "ray_hits_w", None)
    semantic = getattr(scanner_data, "semantic_map", None)
    if ray_hits is None or semantic is None:
        raise RuntimeError("M1 AME geometry reward requires scanner ray_hits_w and semantic_map")
    pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
    resolution = float(getattr(pattern_cfg, "resolution", 0.01))
    terrain = parallelism_terrain_from_scan(
        ray_hits,
        semantic,
        getattr(scanner_data, "valid_mask", None),
        resolution=resolution,
    )
    root_pos = finite_safe_tensor(root_pos, lower=-1.0e4, upper=1.0e4)
    root_quat = finite_safe_quaternion_wxyz(robot.data.root_quat_w)
    safe_joint_pos = finite_safe_tensor(robot.data.joint_pos, lower=-4.0, upper=4.0)
    collision = finite_safe_tensor(_m1_geometry_collision_event(
        root_pos_w=root_pos,
        root_quat_w=root_quat,
        joint_pos=safe_joint_pos,
        terrain=terrain,
    ), lower=0.0, upper=1.0)
    invalid = getattr(env, "_m1_invalid_state_last", None)
    if invalid is not None:
        collision = torch.where(invalid.to(device=collision.device), torch.zeros_like(collision), collision)
    return collision


@configclass
class M1AmeRewardsCfg(CrossLargeComplexPpoRewardsCfg):
    """Go2 pure-PPO reward weights with M1 selectors and wheel semantics."""

    joint_vel = RewTerm(
        func=m1_joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": M1_LEG_JOINT_CFG},
    )
    joint_acc = RewTerm(
        func=m1_joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": M1_ALL_JOINT_CFG},
    )
    joint_torques = RewTerm(
        func=m1_joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": M1_ALL_JOINT_CFG},
    )
    dof_pos_limits = RewTerm(
        func=m1_joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": M1_LEG_JOINT_CFG},
    )
    energy = RewTerm(
        func=m1_energy,
        weight=-2.0e-5,
        params={"asset_cfg": M1_ALL_JOINT_CFG},
    )
    joint_pos = RewTerm(
        func=m1_joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": M1_LEG_JOINT_CFG,
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    track_lin_vel_xy = RewTerm(
        func=m1_track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=m1_track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    base_linear_velocity = RewTerm(func=m1_base_linear_velocity, weight=-2.0)
    base_angular_velocity = RewTerm(func=m1_base_angular_velocity, weight=-0.05)
    action_rate = RewTerm(func=m1_action_rate, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=m1_flat_orientation, weight=-2.5)
    feet_air_time = RewTerm(
        func=m1_feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": M1_SUPPORT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=m1_air_time_variance,
        weight=-1.0,
        params={"sensor_cfg": M1_SUPPORT_SENSOR_CFG},
    )
    feet_slide = RewTerm(
        func=m1_feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": M1_SUPPORT_ASSET_CFG,
            "sensor_cfg": M1_SUPPORT_SENSOR_CFG,
        },
    )
    undesired_contacts = RewTerm(
        func=m1_undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=list(M1_UNDESIRED_CONTACT_BODY_NAMES),
            ),
        },
    )
    parallelism_geometry_collision = None
    m1_geometry_collision = RewTerm(
        func=m1_geometry_collision_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "scanner_cfg": SceneEntityCfg("semantic_height_scanner"),
        },
    )
    fixed_gait_contact_match = RewTerm(
        func=m1_fixed_gait_contact_match,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "cycle_steps": 24,
            "half_cycle_steps": 12,
            "standstill_threshold": 0.05,
            "contact_force_threshold": 1.0,
            "sensor_cfg": M1_SUPPORT_SENSOR_CFG,
        },
    )


@configclass
class M1AmeTerminationsCfg:
    """M1 terminations without planner-reference conditions."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=isaac_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["BASE_LINK"]),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(func=go2_mdp.bad_orientation, params={"limit_angle": 0.8})
    invalid_state = DoneTerm(
        func=m1_termination_invalid_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "leg_joint_cfg": SceneEntityCfg("robot", joint_names=list(M1_LEG_JOINT_NAMES)),
            "wheel_joint_cfg": SceneEntityCfg("robot", joint_names=list(M1_WHEEL_JOINT_NAMES)),
        },
    )


__all__ = [
    "M1AmeRewardsCfg",
    "M1AmeTerminationsCfg",
    "M1_ALL_JOINT_NAMES",
    "M1_LEG_JOINT_NAMES",
    "M1_SUPPORT_BODY_NAMES",
    "M1_UNDESIRED_CONTACT_BODY_NAMES",
    "M1_WHEEL_JOINT_NAMES",
    "m1_invalid_state_mask",
    "m1_geometry_collision_penalty",
]
