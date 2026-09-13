from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from extension.parallelism.config import ParallelismCfg
from extension.parallelism.kinematics import HIP_OFFSETS, LEG_SIDE_SIGNS
from extension.parallelism.root import RootRollout, clamp_command
from extension.parallelism.robot_backend import RobotBackend, get_robot_backend
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
    robot_backend: RobotBackend | None = None,
) -> CandidateSet:
    backend = robot_backend or get_robot_backend("go2")
    root_pos = root.root_pos_w
    root_rpy = root.root_rpy_w
    batch = int(root_pos.shape[0])
    joint = torch.as_tensor(state.joint_pos, dtype=root_pos.dtype, device=root_pos.device)
    joint_ref = joint[:, None].expand(-1, int(cfg.horizon), -1)
    geometry = backend.fk(
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
    if str(backend.name).lower() == "m1":
        frame_ref = torch.where(
            (leg_index == 0) | (leg_index == 3),
            torch.full_like(leg_index, max(int(cfg.half_cycle) - 1, 0)),
            torch.full_like(leg_index, max(int(cfg.horizon) - 1, 0)),
        )
    hip_ref = hip[:, frame_ref, leg_index]
    yaw_ref = root_rpy[:, frame_ref, 2]
    if str(backend.name).lower() == "m1":
        # M1 has a different hip layout; derive the planar offsets from its FK.
        root_ref_xy = root.root_pos_w[:, frame_ref, :2]
        yaw_cos = torch.cos(yaw_ref)
        yaw_sin = torch.sin(yaw_ref)
        hip_delta_w = hip_ref[..., :2] - root_ref_xy
        hip_offset_body = torch.stack(
            (
                yaw_cos * hip_delta_w[..., 0] + yaw_sin * hip_delta_w[..., 1],
                -yaw_sin * hip_delta_w[..., 0] + yaw_cos * hip_delta_w[..., 1],
            ),
            dim=-1,
        )
        default_side = root_pos.new_tensor((1.0, -1.0, 1.0, -1.0)).view(1, 4)
        side = torch.where(
            hip_offset_body[..., 1].abs() > 1.0e-6,
            hip_offset_body[..., 1].sign(),
            default_side,
        )
    else:
        # Preserve the original Go2 candidate geometry and scoring exactly.
        side = torch.tensor(LEG_SIDE_SIGNS, dtype=root_pos.dtype, device=root_pos.device)
        hip_offset_body = torch.tensor(
            HIP_OFFSETS, dtype=root_pos.dtype, device=root_pos.device
        )[:, :2]
    lateral_bias_body = torch.stack(
        (
            torch.zeros_like(side),
            side * float(cfg.hip_lateral_bias_m),
        ),
        dim=-1,
    )
    lateral_bias_w = _yaw_rotate(
        lateral_bias_body
        if str(backend.name).lower() == "m1"
        else lateral_bias_body.view(1, 4, 2),
        yaw_ref,
    )
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
    candidate_z = height + float(cfg.foot_contact_offset_m)
    candidate = torch.cat((candidate_xy, candidate_z[..., None]), dim=-1)
    command = clamp_command(torch.as_tensor(command_body, dtype=root_pos.dtype, device=root_pos.device), cfg)
    period = float(cfg.half_cycle) * float(cfg.dt)
    dtheta = command[:, 2] * period
    hip_offset = hip_offset_body
    c = torch.cos(dtheta)[:, None]
    s = torch.sin(dtheta)[:, None]
    rot_x = c * hip_offset[..., 0] - s * hip_offset[..., 1]
    rot_y = s * hip_offset[..., 0] + c * hip_offset[..., 1]
    yaw_disp = torch.stack((rot_x, rot_y), dim=-1) - hip_offset
    target_body = command[:, None, :2] * period * float(cfg.foothold_step_gain) + yaw_disp
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
