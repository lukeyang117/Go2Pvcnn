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
M1_WHEEL_BODY_CFG = SceneEntityCfg("robot", body_names=list(M1_SUPPORT_BODY_NAMES))
M1_SUPPORT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=list(M1_SUPPORT_BODY_NAMES))
M1_SUPPORT_ASSET_CFG = SceneEntityCfg("robot", body_names=list(M1_SUPPORT_BODY_NAMES))

_M1_PLANNER_INDICES = tuple(M1_ASSET_JOINT_NAMES.index(name) for name in M1_PLANNER_JOINT_NAMES)


def _m1_geometry_collision_event(
    root_pos_w: Tensor,
    root_quat_w: Tensor,
    joint_pos: Tensor,
    terrain: ParallelismTerrain,
) -> Tensor:
    root_pos = torch.as_tensor(root_pos_w, dtype=torch.float32)
    root_quat = torch.as_tensor(root_quat_w, dtype=root_pos.dtype, device=root_pos.device)
    indices = torch.as_tensor(_M1_PLANNER_INDICES, dtype=torch.long, device=root_pos.device)
    planner_joint = torch.as_tensor(joint_pos, dtype=root_pos.dtype, device=root_pos.device).index_select(-1, indices)
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
    return _m1_geometry_collision_event(
        root_pos_w=root_pos,
        root_quat_w=robot.data.root_quat_w,
        joint_pos=robot.data.joint_pos,
        terrain=terrain,
    )


@configclass
class M1AmeRewardsCfg(CrossLargeComplexPpoRewardsCfg):
    """Go2 pure-PPO reward weights with M1 selectors and wheel semantics."""

    joint_vel = RewTerm(
        func=go2_mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": M1_LEG_JOINT_CFG},
    )
    joint_acc = RewTerm(
        func=go2_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": M1_ALL_JOINT_CFG},
    )
    joint_torques = RewTerm(
        func=go2_mdp.joint_torques_l2,
        weight=-2.0e-4,
        params={"asset_cfg": M1_ALL_JOINT_CFG},
    )
    dof_pos_limits = RewTerm(
        func=go2_mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": M1_LEG_JOINT_CFG},
    )
    energy = RewTerm(
        func=go2_mdp.energy,
        weight=-2.0e-5,
        params={"asset_cfg": M1_ALL_JOINT_CFG},
    )
    joint_pos = RewTerm(
        func=go2_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": M1_LEG_JOINT_CFG,
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    feet_air_time = RewTerm(
        func=go2_mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": M1_SUPPORT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    air_time_variance = RewTerm(
        func=go2_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": M1_SUPPORT_SENSOR_CFG},
    )
    feet_slide = RewTerm(
        func=go2_mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": M1_SUPPORT_ASSET_CFG,
            "sensor_cfg": M1_SUPPORT_SENSOR_CFG,
        },
    )
    undesired_contacts = RewTerm(
        func=isaac_mdp.undesired_contacts,
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
        func=tracking_mdp.fixed_gait_contact_match_reward,
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


__all__ = [
    "M1AmeRewardsCfg",
    "M1AmeTerminationsCfg",
    "M1_ALL_JOINT_NAMES",
    "M1_LEG_JOINT_NAMES",
    "M1_SUPPORT_BODY_NAMES",
    "M1_UNDESIRED_CONTACT_BODY_NAMES",
    "M1_WHEEL_JOINT_NAMES",
    "m1_geometry_collision_penalty",
]
