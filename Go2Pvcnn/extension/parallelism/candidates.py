from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from extension.parallelism.config import ParallelismCfg
from extension.parallelism.kinematics import HIP_OFFSETS, LEG_SIDE_SIGNS, fk_go2
from extension.parallelism.root import RootRollout, clamp_command
from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismState, ParallelismTerrain


@dataclass(frozen=True)
class CandidateSet:
    candidate_w: Tensor
    offset_body: Tensor
    hip_ref_w: Tensor
    candidate_center_w: Tensor
    yaw_ref: Tensor
    score_target_body: Tensor
    candidate_valid_map: Tensor
    candidate_semantic: Tensor


def _disk_offsets(cfg: ParallelismCfg, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    idx = torch.arange(int(cfg.candidates_per_leg), dtype=dtype, device=device)
    radius = float(cfg.candidate_radius_m) * torch.sqrt((idx + 0.5) / float(cfg.candidates_per_leg))
    theta = idx * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)


def _yaw_rotate(offset: Tensor, yaw: Tensor) -> Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x = cosine * offset[..., 0] - sine * offset[..., 1]
    y = sine * offset[..., 0] + cosine * offset[..., 1]
    return torch.stack((x, y), dim=-1)


def build_candidates(
    root: RootRollout,
    state: ParallelismState,
    command_body: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
) -> CandidateSet:
    root_pos = root.root_pos_w
    root_rpy = root.root_rpy_w
    batch = int(root_pos.shape[0])
    joint = torch.as_tensor(state.joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    joint_ref = joint[:, None].expand(-1, int(cfg.horizon), -1)
    geometry = fk_go2(
        root_pos.reshape(batch * int(cfg.horizon), 3),
        root_rpy.reshape(batch * int(cfg.horizon), 3),
        joint_ref.reshape(batch * int(cfg.horizon), 12),
    )
    hip = geometry.hip_pos_w.reshape(batch, int(cfg.horizon), 4, 3)
    leg_index = torch.arange(4, device=root_pos.device)
    frame_ref = torch.where(
        (leg_index == 0) | (leg_index == 3),
        torch.zeros_like(leg_index),
        torch.full_like(leg_index, int(cfg.half_cycle)),
    )
    hip_ref = hip[:, frame_ref, leg_index]
    yaw_ref = root_rpy[:, frame_ref, 2]
    side = torch.tensor(LEG_SIDE_SIGNS, dtype=root_pos.dtype, device=root_pos.device)
    lateral_bias_body = torch.stack(
        (
            torch.zeros_like(side),
            side * float(cfg.hip_lateral_bias_m),
        ),
        dim=-1,
    )
    lateral_bias_w = _yaw_rotate(lateral_bias_body.view(1, 4, 2), yaw_ref)
    candidate_center = torch.cat(
        (
            hip_ref[..., :2] + lateral_bias_w,
            hip_ref[..., 2:3],
        ),
        dim=-1,
    )
    offsets = _disk_offsets(cfg, dtype=root_pos.dtype, device=root_pos.device)
    offset_w = _yaw_rotate(offsets.view(1, 1, int(cfg.candidates_per_leg), 2), yaw_ref[:, :, None])
    candidate_xy = candidate_center[:, :, None, :2] + offset_w
    flat_xy = candidate_xy.reshape(batch, 4 * int(cfg.candidates_per_leg), 2)
    query = query_height_semantic_valid(terrain, flat_xy)
    height = query.height.reshape(batch, 4, int(cfg.candidates_per_leg))
    semantic = query.semantic.reshape(batch, 4, int(cfg.candidates_per_leg))
    valid = query.valid.reshape(batch, 4, int(cfg.candidates_per_leg))
    candidate = torch.cat((candidate_xy, height[..., None]), dim=-1)
    command = clamp_command(torch.as_tensor(command_body, dtype=root_pos.dtype, device=root_pos.device), cfg)
    period = float(cfg.half_cycle) * float(cfg.dt)
    dtheta = command[:, 2] * period
    hip_offset = torch.tensor(HIP_OFFSETS, dtype=root_pos.dtype, device=root_pos.device)[:, :2]
    c = torch.cos(dtheta)[:, None]
    s = torch.sin(dtheta)[:, None]
    rot_x = c * hip_offset[None, :, 0] - s * hip_offset[None, :, 1]
    rot_y = s * hip_offset[None, :, 0] + c * hip_offset[None, :, 1]
    yaw_disp = torch.stack((rot_x, rot_y), dim=-1) - hip_offset[None]
    target_body = command[:, None, :2] * period + yaw_disp
    return CandidateSet(
        candidate_w=candidate,
        offset_body=offsets,
        hip_ref_w=hip_ref,
        candidate_center_w=candidate_center,
        yaw_ref=yaw_ref,
        score_target_body=target_body,
        candidate_valid_map=valid,
        candidate_semantic=semantic,
    )
