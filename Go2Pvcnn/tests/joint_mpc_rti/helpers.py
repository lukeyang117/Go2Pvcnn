from __future__ import annotations

import torch


def make_state(batch: int, *, device: str = "cpu", dtype: torch.dtype = torch.float32):
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.types import JointMpcRtiState

    root_pos = torch.zeros(batch, 3, device=device, dtype=dtype)
    root_pos[:, 2] = JointMpcRtiCfg().loss_terms.posture_root_clearance
    joint = torch.tensor([0.0, 0.8, -1.5] * 4, device=device, dtype=dtype).expand(batch, -1).clone()
    return JointMpcRtiState(
        root_pos_w=root_pos,
        root_rpy_w=torch.zeros(batch, 3, device=device, dtype=dtype),
        joint_pos=joint,
        root_lin_vel_b=torch.zeros(batch, 3, device=device, dtype=dtype),
        root_ang_vel_b=torch.zeros(batch, 3, device=device, dtype=dtype),
        joint_vel=torch.zeros(batch, 12, device=device, dtype=dtype),
    )


def make_command(
    batch: int,
    *,
    vx: float = 0.2,
    vy: float = 0.0,
    yaw: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    return torch.tensor([vx, vy, yaw], dtype=torch.float32, device=device).expand(batch, -1).clone()


def make_flat_field(batch: int, *, device: str = "cpu"):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    return build_field_batch(
        height_w=torch.zeros(batch, 151, 151, device=device),
        semantic_id=torch.zeros(batch, 151, 151, dtype=torch.long, device=device),
        origin_w=torch.zeros(batch, 3, device=device),
        yaw_w=torch.zeros(batch, device=device),
        timestamp=torch.zeros(batch, device=device),
        version=torch.ones(batch, dtype=torch.long, device=device),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
