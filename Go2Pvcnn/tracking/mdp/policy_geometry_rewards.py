"""Planner-free geometry rewards for the live policy pose."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

try:
    from isaaclab.managers import SceneEntityCfg
except Exception:  # noqa: BLE001 - keeps tensor tests independent of IsaacLab imports.
    class SceneEntityCfg:  # type: ignore[no-redef]
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            for key, value in kwargs.items():
                setattr(self, key, value)

from extension.convention import extract_roll_pitch_batch, extract_yaw_batch
from extension.parallelism.collision import official_collision_mask
from extension.parallelism.config import ParallelismCfg
from extension.parallelism.kinematics import Go2ParallelGeometry, fk_go2
from extension.parallelism.types import ParallelismTerrain


_PLANNER_JOINT_ORDER = (
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
_COLLISION_CFG = ParallelismCfg()


def _normalize_name(name: str) -> str:
    value = str(name).split("/")[-1]
    return value.split(":")[-1].lower()


def _order_indices(
    source_order: Sequence[str],
    target_order: Sequence[str],
    *,
    device: torch.device,
) -> Tensor | None:
    source_to_index = {_normalize_name(name): index for index, name in enumerate(source_order)}
    indices: list[int] = []
    for name in target_order:
        index = source_to_index.get(_normalize_name(name))
        if index is None:
            return None
        indices.append(index)
    return torch.tensor(indices, dtype=torch.long, device=device)


def _reorder_joint_to_planner(joint_pos: Tensor, joint_names: Sequence[str] | None) -> Tensor:
    values = torch.as_tensor(joint_pos)
    if not joint_names or int(values.shape[-1]) != len(tuple(joint_names)):
        return values
    indices = _order_indices(tuple(joint_names), _PLANNER_JOINT_ORDER, device=values.device)
    if indices is None:
        return values
    return values.index_select(-1, indices)


def parallelism_terrain_from_scan(
    ray_hits_w: Tensor,
    semantic_map: Tensor,
    valid_mask: Tensor | None,
    *,
    resolution: float,
) -> ParallelismTerrain:
    """Build the static terrain object required by official collision checks."""

    hits = torch.as_tensor(ray_hits_w, dtype=torch.float32)
    if hits.ndim != 3 or int(hits.shape[-1]) != 3:
        raise ValueError(f"ray_hits_w must have shape [B,H*W,3], got {tuple(hits.shape)}")
    batch, ray_count, _ = hits.shape
    side = int(round(float(ray_count) ** 0.5))
    if side * side != int(ray_count):
        raise ValueError(f"scanner ray count {ray_count} is not a square grid")

    grid = hits.reshape(batch, side, side, 3)
    semantic = torch.as_tensor(semantic_map, dtype=torch.long, device=hits.device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0).expand(batch, -1, -1)
    if tuple(semantic.shape) != (batch, side, side):
        raise ValueError(
            f"semantic_map must have shape {(batch, side, side)}, got {tuple(semantic.shape)}"
        )

    if valid_mask is None:
        valid = torch.isfinite(grid).all(dim=-1)
    else:
        valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=hits.device)
        if valid.ndim == 2:
            valid = valid.unsqueeze(0).expand(batch, -1, -1)
        if tuple(valid.shape) != (batch, side, side):
            raise ValueError(
                f"valid_mask must have shape {(batch, side, side)}, got {tuple(valid.shape)}"
            )

    origin = torch.zeros(batch, 3, dtype=hits.dtype, device=hits.device)
    origin[:, :2] = grid[:, 0, 0, :2]
    if side > 1:
        step_xy = grid[:, 0, 1, :2] - grid[:, 0, 0, :2]
        yaw = torch.atan2(step_xy[:, 1], step_xy[:, 0])
    else:
        yaw = torch.zeros(batch, dtype=hits.dtype, device=hits.device)

    return ParallelismTerrain(
        height_w=torch.nan_to_num(grid[..., 2], nan=0.0, posinf=0.0, neginf=0.0),
        semantic_id=semantic,
        valid_mask=valid,
        origin_w=origin,
        yaw_w=yaw,
        resolution=float(resolution),
    )


def _terrain_from_scanner(scanner, root_pos_w: Tensor, *, resolution: float) -> ParallelismTerrain:
    data = scanner.data
    ray_hits_w = getattr(data, "ray_hits_w", None)
    semantic_map = getattr(data, "semantic_map", None)
    valid_mask = getattr(data, "valid_mask", None)
    if semantic_map is None:
        raise RuntimeError("semantic_height_scanner.data.semantic_map is required")
    if ray_hits_w is not None:
        return parallelism_terrain_from_scan(
            ray_hits_w,
            semantic_map,
            valid_mask,
            resolution=resolution,
        )

    elevation_map = getattr(data, "elevation_map", None)
    if elevation_map is None:
        raise RuntimeError("semantic_height_scanner requires ray_hits_w or elevation_map")
    height = torch.as_tensor(elevation_map, dtype=torch.float32, device=root_pos_w.device)
    if height.ndim != 3:
        raise ValueError(f"elevation_map must have shape [B,H,W], got {tuple(height.shape)}")
    batch, height_size, width = height.shape
    if height_size != width:
        raise ValueError(f"elevation_map must be square, got {tuple(height.shape)}")
    side = int(height_size)
    semantic = torch.as_tensor(semantic_map, dtype=torch.long, device=height.device)
    if semantic.ndim == 2:
        semantic = semantic.unsqueeze(0).expand(batch, -1, -1)
    valid = (
        torch.isfinite(height)
        if valid_mask is None
        else torch.as_tensor(valid_mask, dtype=torch.bool, device=height.device)
    )
    half_extent = 0.5 * float(side - 1) * float(resolution)
    origin = torch.zeros(batch, 3, dtype=height.dtype, device=height.device)
    origin[:, 0] = root_pos_w[:, 0] - half_extent
    origin[:, 1] = root_pos_w[:, 1] - half_extent
    return ParallelismTerrain(
        height_w=torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0),
        semantic_id=semantic,
        valid_mask=valid,
        origin_w=origin,
        yaw_w=torch.zeros(batch, dtype=height.dtype, device=height.device),
        resolution=float(resolution),
    )


def _expand_geometry_for_collision(geometry: Go2ParallelGeometry) -> Go2ParallelGeometry:
    batch, leg_count = geometry.foot_pos_w.shape[:2]

    def expand_pose(value: Tensor) -> Tensor:
        return value[:, None, None].expand(batch, leg_count, 1, *value.shape[1:])

    def expand_samples(value: Tensor) -> Tensor:
        return value[:, None, None].expand(batch, leg_count, 1, *value.shape[1:])

    return Go2ParallelGeometry(
        hip_pos_w=expand_pose(geometry.hip_pos_w),
        foot_pos_w=expand_pose(geometry.foot_pos_w),
        knee_pos_w=expand_pose(geometry.knee_pos_w),
        calf_samples_w=expand_samples(geometry.calf_samples_w),
        thigh_samples_w=expand_samples(geometry.thigh_samples_w),
        thigh_pos_w=expand_pose(geometry.thigh_pos_w),
        thigh_rot_w=expand_pose(geometry.thigh_rot_w),
        calf_pos_w=expand_pose(geometry.calf_pos_w),
        calf_rot_w=expand_pose(geometry.calf_rot_w),
        foot_rot_w=expand_pose(geometry.foot_rot_w),
    )


def live_policy_geometry_collision_event(
    root_pos_w: Tensor,
    root_quat_w: Tensor,
    joint_pos: Tensor,
    joint_names: Sequence[str] | None,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg = _COLLISION_CFG,
) -> Tensor:
    """Return one collision event per environment for the live policy pose."""

    root_pos = torch.as_tensor(root_pos_w, dtype=torch.float32)
    root_quat = torch.as_tensor(root_quat_w, dtype=root_pos.dtype, device=root_pos.device)
    joint = _reorder_joint_to_planner(joint_pos, joint_names).to(dtype=root_pos.dtype, device=root_pos.device)
    roll, pitch = extract_roll_pitch_batch(root_quat)
    yaw = extract_yaw_batch(root_quat)
    root_rpy = torch.stack((roll, pitch, yaw), dim=-1)
    geometry = fk_go2(root_pos, root_rpy, joint, capsule_samples=int(cfg.capsule_samples))
    expanded_geometry = _expand_geometry_for_collision(geometry)
    _, collision_bits = official_collision_mask(terrain, expanded_geometry, cfg)
    return collision_bits.any(dim=(1, 2, 3)).to(dtype=torch.float32)


def policy_geometry_collision_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scanner_cfg: SceneEntityCfg = SceneEntityCfg("semantic_height_scanner"),
) -> Tensor:
    """Return the raw live-policy geometry collision event.

    The negative sign belongs to the RewardTerm weight. No reference trajectory,
    contact phase, or Parallelism manager is consulted here.
    """

    robot = env.scene[asset_cfg.name]
    scanner = env.scene[scanner_cfg.name]
    default_device = torch.as_tensor(robot.data.root_pos_w).device
    device = torch.device(getattr(env, "device", default_device))
    root_pos_w = torch.as_tensor(robot.data.root_pos_w, dtype=torch.float32, device=device)
    pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
    resolution = float(getattr(pattern_cfg, "resolution", 0.01))
    terrain = _terrain_from_scanner(scanner, root_pos_w, resolution=resolution)
    return live_policy_geometry_collision_event(
        root_pos_w=root_pos_w,
        root_quat_w=robot.data.root_quat_w,
        joint_pos=robot.data.joint_pos,
        joint_names=tuple(getattr(robot, "joint_names", ())),
        terrain=terrain,
    )


__all__ = [
    "live_policy_geometry_collision_event",
    "parallelism_terrain_from_scan",
    "policy_geometry_collision_penalty",
]
