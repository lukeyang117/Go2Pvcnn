from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from extension.parallelism.collision import build_official_surface_points_l, official_collision_mask
from extension.parallelism.config import OfficialCollisionShapeSpec, ParallelismCfg
from extension.parallelism.ik import ik_go2
from extension.parallelism.kinematics import JOINT_LOWER, JOINT_UPPER, fk_go2
from extension.parallelism.m1_collision import build_m1_surface_points_l
from extension.parallelism.m1_kinematics import (
    M1_ASSET_JOINT_NAMES,
    M1_CFG,
    M1_PLANNER_JOINT_NAMES,
    M1_WHEEL_JOINT_NAMES,
    m1_fk,
    m1_ik,
    m1_joint_limit_mask,
)


@dataclass(frozen=True)
class RobotBackend:
    name: str
    planner_joint_names: tuple[str, ...]
    asset_joint_names: tuple[str, ...]
    wheel_joint_names: tuple[str, ...]
    support_body_names: tuple[str, ...]
    cfg: ParallelismCfg
    fk: Callable
    ik: Callable
    joint_limit_mask: Callable[[Tensor], Tensor]
    collision_shapes: tuple[OfficialCollisionShapeSpec, ...]
    surface_points_builder: Callable
    collision_mask: Callable


def _go2_joint_limit_mask(joint_pos: Tensor, _leg_indices: Tensor | None = None) -> Tensor:
    lower = joint_pos.new_tensor(JOINT_LOWER)
    upper = joint_pos.new_tensor(JOINT_UPPER)
    return ((joint_pos >= lower) & (joint_pos <= upper)).all(dim=-1)


def _go2_backend() -> RobotBackend:
    cfg = ParallelismCfg()
    return RobotBackend(
        name="go2",
        planner_joint_names=tuple(f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("abad", "hip", "knee")),
        asset_joint_names=(),
        wheel_joint_names=(),
        support_body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
        cfg=cfg,
        fk=fk_go2,
        ik=ik_go2,
        joint_limit_mask=_go2_joint_limit_mask,
        collision_shapes=tuple(cfg.official_collision_shapes),
        surface_points_builder=build_official_surface_points_l,
        collision_mask=official_collision_mask,
    )


def _m1_backend() -> RobotBackend:
    return RobotBackend(
        name="m1",
        planner_joint_names=M1_PLANNER_JOINT_NAMES,
        asset_joint_names=M1_ASSET_JOINT_NAMES,
        wheel_joint_names=M1_WHEEL_JOINT_NAMES,
        support_body_names=("FBL_FOOT_LINK", "FAR_FOOT_LINK", "RBL_FOOT_LINK", "RAR_FOOT_LINK"),
        cfg=M1_CFG,
        fk=m1_fk,
        ik=m1_ik,
        joint_limit_mask=m1_joint_limit_mask,
        collision_shapes=tuple(M1_CFG.official_collision_shapes),
        surface_points_builder=build_m1_surface_points_l,
        collision_mask=lambda terrain, geometry, cfg: official_collision_mask(
            terrain,
            geometry,
            cfg,
            surface_points_builder=build_m1_surface_points_l,
        ),
    )


def get_robot_backend(name: str = "go2") -> RobotBackend:
    normalized = str(name).strip().lower()
    if normalized == "go2":
        return _go2_backend()
    if normalized == "m1":
        return _m1_backend()
    raise ValueError(f"unknown robot backend {name!r}; expected one of: go2, m1")
