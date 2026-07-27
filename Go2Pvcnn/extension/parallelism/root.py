from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from extension.parallelism.config import ParallelismCfg
from extension.parallelism.kinematics import fk_go2
from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismState, ParallelismTerrain


@dataclass(frozen=True)
class RootRollout:
    root_pos_w: Tensor
    root_rpy_w: Tensor
    clamped_command_body: Tensor


def clamp_command(command_body: Tensor, cfg: ParallelismCfg) -> Tensor:
    command = torch.as_tensor(command_body)
    limits = torch.tensor(
        [cfg.vx_limit, cfg.vy_limit, cfg.vyaw_limit],
        dtype=command.dtype,
        device=command.device,
    )
    return torch.maximum(torch.minimum(command, limits), -limits)


def _half_profile(cfg: ParallelismCfg, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    tau = torch.linspace(0.0, 1.0, int(cfg.half_cycle), dtype=dtype, device=device)
    return tau * tau * (3.0 - 2.0 * tau)


def rollout_root(
    state: ParallelismState,
    command_body: Tensor,
    terrain: ParallelismTerrain,
    cfg: ParallelismCfg,
) -> RootRollout:
    root0 = torch.as_tensor(state.root_pos_w)
    rpy0 = torch.as_tensor(state.root_rpy_w, dtype=root0.dtype, device=root0.device)
    command = clamp_command(torch.as_tensor(command_body, dtype=root0.dtype, device=root0.device), cfg)
    half = _half_profile(cfg, dtype=root0.dtype, device=root0.device)
    disp_half = command[:, :2] * (float(cfg.half_cycle) * float(cfg.dt))
    yaw_half = command[:, 2] * (float(cfg.half_cycle) * float(cfg.dt))
    first_xy_body = half[None, :, None] * disp_half[:, None, :]
    second_xy_body = disp_half[:, None, :] + half[None, :, None] * disp_half[:, None, :]
    xy_body = torch.cat((first_xy_body, second_xy_body), dim=1)
    first_yaw = half[None, :] * yaw_half[:, None]
    second_yaw = yaw_half[:, None] + half[None, :] * yaw_half[:, None]
    yaw_delta = torch.cat((first_yaw, second_yaw), dim=1)
    yaw = rpy0[:, 2:3] + yaw_delta
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    world_dx = cosine * xy_body[..., 0] - sine * xy_body[..., 1]
    world_dy = sine * xy_body[..., 0] + cosine * xy_body[..., 1]
    root_xy = root0[:, None, :2] + torch.stack((world_dx, world_dy), dim=-1)
    joint = torch.as_tensor(state.joint_pos, dtype=root0.dtype, device=root0.device)
    foot0 = (
        torch.as_tensor(state.foot_pos_w, dtype=root0.dtype, device=root0.device)
        if state.foot_pos_w is not None
        else fk_go2(root0, rpy0, joint).foot_pos_w
    )
    stance_first = query_height_semantic_valid(terrain, foot0[:, (1, 2), :2]).height.mean(dim=1)
    stance_second = query_height_semantic_valid(terrain, foot0[:, (0, 3), :2]).height.mean(dim=1)
    z = torch.cat(
        (
            stance_first[:, None].expand(-1, int(cfg.half_cycle)),
            stance_second[:, None].expand(-1, int(cfg.half_cycle)),
        ),
        dim=1,
    ) + float(cfg.root_clearance_m)
    root_pos = torch.cat((root_xy, z[..., None]), dim=-1)
    root_rpy = torch.zeros(root_pos.shape[0], int(cfg.horizon), 3, dtype=root0.dtype, device=root0.device)
    root_rpy[..., 0] = rpy0[:, None, 0]
    root_rpy[..., 1] = rpy0[:, None, 1]
    root_rpy[..., 2] = yaw
    return RootRollout(root_pos_w=root_pos, root_rpy_w=root_rpy, clamped_command_body=command)
